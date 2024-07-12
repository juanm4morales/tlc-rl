import sys
import os
import traci
from sumolib import checkBinary
from typing import Dict
import sys
import gymnasium as gym

from tscRL.util.discrete import Discrete

import random

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
    

state_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.states')
os.makedirs(state_dir, exist_ok=True)
HALTED = "halted"
WAITING_TIME = "waitingTime"

class Vehicle:
    def __init__(self, position, length, maxSpeed):
        self.position = position
        self.length = length
        self.maxSpeed = maxSpeed
        self.waitingTime = 0
    
    def update(self, deltaTime):
        self.waitingTime += deltaTime
        
class Lane:
    def __init__(self, laneId, laneLength, edge=False):
        # self.vehicleMinGap = vehicleMinGap
        # self.vehicles = []
        self.laneId = laneId
        # self.laneLength = laneLength
        self.lastStepHaltedVehicles = 0
        self.lastStepWaitingTime = 0
        self.edge = edge
        
    def update(self):
        if (self.edge):
            self.lastStepHaltedVehicles = traci.edge.getLastStepHaltingNumber(self.laneId)
            self.lastStepWaitingTime = traci.edge.getWaitingTime(self.laneId)
        else:
            self.lastStepHaltedVehicles = traci.lane.getLastStepHaltingNumber(self.laneId)
            self.lastStepWaitingTime = traci.lane.getWaitingTime(self.laneId)  

class TrafficLight:
    class Phase:
        def __init__(self, state, yellowTransition):
            self.state = state
            self.yellowTransition = yellowTransition
            
    # n=north, s=south, e=east, w=weast, l=left turn allowed (without priority)
    # G=Green with priority, g=green without priority, y=yellow, r=red
    
    
    def __init__(self, id, initialPhase, yellowTime, minGreenTime, phases = None):
        self.id = id
        self.currentPhase = "init"
        self.nextPhase = initialPhase
        self.yellowTime = yellowTime
        self.minGreenTime = minGreenTime

        self.yellow = False
        self.currentPhaseTime = 0
        
        if phases == None:
            self.PHASES = phases
            
    def update(self):
        if (self.currentPhase != self.nextPhase):
            if (self.yellow and self.currentPhaseTime >= self.yellowTime) or not self.yellow:
                nextPhaseState = self.PHASES[self.nextPhase].state
                traci.trafficlight.setRedYellowGreenState(0, nextPhaseState)
                self.currentPhase = self.nextPhase
                self.currentPhaseTime = 0
                self.yellow = False
                
        traci.simulationStep() 
        self.currentPhaseTime += 1
    
    def canChange(self):
        return (
            not(self.yellow)
            and (self.currentPhaseTime >= self.minGreenTime)
            and (self.currentPhaseTime>=self.yellowTime)
            )
    
    def changePhase(self, newPhase):
        if (self.currentPhase != newPhase):
            if self.canChange() or (self.currentPhase == "init"):
                if (self.currentPhase != "init"):  
                    yellowPhaseState = self.PHASES[self.currentPhase].yellowTransition
                    traci.trafficlight.setRedYellowGreenState(0, yellowPhaseState)
                    self.yellow = True
                    
                previousPhaseTime = self.currentPhaseTime
                self.currentPhaseTime = 0
                self.nextPhase = newPhase
                    
                return previousPhaseTime
            else:
                # No ha pasado el mínimo de tiempo de duración de la fase
                return -1
        else:
            # La fase actual es la misma que la nueva
            return -2

class State:
    def __init__(self, tlPhase, lanes: Dict[str, Lane], discreteClass, laneInfo="halted"):
        self.discreteClass = discreteClass
        self.tlPhase = tlPhase
        self.discreteLaneInfo = self.discretizeLaneInfo(lanes, laneInfo)

    def getTupleState(self):
        return (self.tlPhase, *self.discreteLaneInfo)
    
    def discretizeLaneInfo(self, lanes: Dict[str, Lane], laneInfo):
        # discreteLaneQueue: Dict[str, int] = {}
        discreteLaneInfo = []
        if laneInfo == "waitingTime":
            for lane in lanes.values():
                discreteLaneInfo.append(self.discreteClass.log_interval(lane.lastStepWaitingTime))
            return discreteLaneInfo
        else:
            for lane in lanes.values():
                discreteLaneInfo.append(self.discreteClass.log_interval(lane.lastStepHaltedVehicles))
            if laneInfo != "halted":
                print("Warning: " + "Invalid laneInfo value = " + laneInfo + ". \"halted\" value was assigned instead.")
            return discreteLaneInfo          
        
class SumoEnvironment(gym.Env):
    MAX_VEH_LANE = 30      # adjust according to lane length? .. Before 32
    MAX_WAITING_TIME = 500 # Param? .. Before 300
    def __init__(self, sumocfgFile, deltaTime=5, yellowTime=4, minGreenTime=5, gui=False, edges=False, discreteIntervals=6, laneInfo="halted", rewardFn="diff_halted", fixedTL=False, simTime=43800, sumoLog=False):
        self.sumocfgFile = sumocfgFile
        self.stateFile = os.path.join(state_dir, 'initialState.xml')
        self.gui = gui
        if gui:
            self.sumoBinary = checkBinary("sumo-gui")
        else:
            self.sumoBinary = checkBinary("sumo")
        self.simTime = simTime
        assert(yellowTime < deltaTime)
        self.deltaTime = deltaTime
        self.fixedTL=fixedTL
        self.haltedVehicles = 0
        self.waitingTime = 0
        self.cumulativeWaitingTime = 0
        
        if rewardFn in self.rewardFns.keys():
            self.rewardFn = self.rewardFns[rewardFn]
        else:
            self.rewardFn = self.rewardFns["diff_halted"]
            print("Warning: Invalid rewardFn value. \"diff_halted\" value was assigned instead.")
        
        self.sumoLog = sumoLog
        
        self.initializeSimulation()
        
        tls_ids = traci.trafficlight.getIDList()
        
        self.trafficLight = TrafficLight(tls_ids[0], "init", yellowTime, minGreenTime, laneInfo)
                        
        lanesIds = traci.trafficlight.getControlledLanes(tls_ids[0])
        self.lanes: Dict[str, Lane] = {}
        for laneId in lanesIds:
            if "in" in laneId:
                if edges:
                    laneId = traci.lane.getEdgeID(laneId) 
                if laneId not in self.lanes:
                    self.lanes[laneId] = Lane(laneId, traci.lane.getLength(laneId), edge = edges)
        
        if (laneInfo == WAITING_TIME):
            if (edges):
                self.discreteClass = Discrete(discreteIntervals, self.MAX_WAITING_TIME*2)
            else:
                self.discreteClass = Discrete(discreteIntervals, self.MAX_WAITING_TIME)
        else:
            if (edges):
                self.discreteClass = Discrete(discreteIntervals, self.MAX_VEH_LANE*2)
            else:
                self.discreteClass = Discrete(discreteIntervals, self.MAX_VEH_LANE)
                    
        self.warmingUpSimulation()
        
        if self.fixedTL:
            traci.trafficlight.setProgram(tls_ids[0], "2")
        else:
            traci.trafficlight.setProgram(tls_ids[0], "0")

    @property
    def simStep(self):
        return traci.simulation.getTime()
    
    @property
    def actionSpace(self):
        return [actionKey for actionKey in self.trafficLight.PHASES if actionKey != 'init']
       
    def initializeSimulation(self):
        sumoCMD = [self.sumoBinary, "-c", self.sumocfgFile
                   #"--tripinfo-output", "tripinfo.xml"
                   ]
        
        if not(self.sumoLog):
            sumoCMD.append("--no-step-log")
            sumoCMD.append("--no-warnings")
        if self.gui:
            sumoCMD.append("-S")
        
        try:
            traci.start(sumoCMD)
            
        except traci.TraCIException as traci_e:
            print(traci_e, end="")
            print(" Closing connection...")
            traci.close()
            import time
            time.sleep(1)
            print(traci_e, end="")
            print(" Starting new connection...")
            traci.start(sumoCMD)
            print("OK")
    
    def warmingUpSimulation(self):
        traci.simulationStep(599) # Warming Time
        traci.trafficlight.setRedYellowGreenState(0, self.trafficLight.PHASES["init"].state)
        traci.simulationStep()
        self.waitingTime = self.getTotalWaitingTime()
        self.haltedVehicles = self.getTotalHaltedVehicles()
        self.trafficLight.currentPhase = "init"
        self.trafficLight.nextPhase = "init"
        for lane in self.lanes.values():
            lane.update()
        traci.simulation.saveState(self.stateFile)
        
    def setTLProgram(self, programID: int):
        try:
            traci.trafficlight.setProgram(tls_ids[0], programID)
        except traci.TraCIException as traci_e:
            print(traci_e, end="")
            print(" Program ID setted to 1.")
            traci.trafficlight.setProgram(tls_ids[0], "1")
            
    def getCurrentState(self):
        state = State(self.trafficLight.currentPhase, self.lanes, self.discreteClass)
        return state.getTupleState()
    
    def getTotalHaltedVehicles(self):
        return sum(lane.lastStepHaltedVehicles for lane in self.lanes.values())
    
    def getTotalWaitingTime(self):
        return sum(lane.lastStepWaitingTime for lane in self.lanes.values())
    
    def computeReward(self):
        return self.rewardFn(self)
    
    def getAccumulatedWaitingTime(self):
        accumulatedWaitingTime = 0
        for lane in self.lanes.values():
            vehicles = traci.lane.getLastStepVehicleIDs(lane.laneId)
            for vehicle in vehicles:
                accumulatedWaitingTime += traci.vehicle.getAccumulatedWaitingTime(vehicle)
        return accumulatedWaitingTime
    
    def diffHalted(self):
        currentStepHaltedVehicles = self.getTotalHaltedVehicles()
        reward = self.haltedVehicles-currentStepHaltedVehicles
        self.haltedVehicles = currentStepHaltedVehicles
        return reward

    def diffWaitingTime(self):
        currentWaitingTime = self.getTotalWaitingTime()
        reward = self.waitingTime - currentWaitingTime
        self.waitingTime = currentWaitingTime
        return reward

    def diffAccumulatedWaitingTime(self):
        currentAccWaitingTime = self.getAccumulatedWaitingTime()
        reward = self.cumulativeWaitingTime - currentAccWaitingTime
        self.cumulativeWaitingTime = currentAccWaitingTime
        return reward

    def getInfo(self):        
        vehicleCount = traci.vehicle.getIDCount()
        
        if (self.rewardFn in [self.rewardFns["diff_halted"], self.rewardFns["diff_cumulativeWaitingTime"]]):
            self.waitingTime = self.getTotalWaitingTime()
        elif (self.rewardFn == self.rewardFns["diff_waitingTime"]):
            self.haltedVehicles = self.getTotalHaltedVehicles()
            
        meanWaitingTime = self.waitingTime / vehicleCount if vehicleCount > 0 else 0
        # vehicles = traci.vehicle.getIDList()
        # waiting_times = [traci.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]
        info = {
            "sim_step": self.simStep,
            "mean_waiting_time": meanWaitingTime
        }
        return info
        
        
    def step(self, action=None):
        # previousPhaseTime = 0
        # TOMAR ACCIÓN
        if (self.fixedTL):
            for _ in range(self.deltaTime):
                traci.simulationStep()
        else:
            self.trafficLight.changePhase(action)
            # PASO DE TIEMPO (deltaTime)   
            for _ in range(self.deltaTime):
                self.trafficLight.update()
            
        for lane in self.lanes.values():
            lane.update()
        
        state = self.getCurrentState()
        # print(state)
        # print(action)
        reward = self.computeReward()
        
    
        done = traci.simulation.getMinExpectedNumber() == 0 or traci.simulation.getTime() > self.simTime

        info = self.getInfo()
        return state, reward, done, info
        
    def reset(self):
        self.trafficLight.yellow = False
        self.trafficLight.currentPhase = "init"
        self.trafficLight.nextPhase = "init"
        self.trafficLight.currentPhaseTime = 0

        self.haltedVehicles = 0
        self.waitingTime = 0
        self.cumulativeWaitingTime = 0
        
        for laneKey in self.lanes:
            self.lanes[laneKey].lastStepHaltedVehicles = 0
            self.lanes[laneKey].waitingTime = 0
            
        traci.simulation.loadState(self.stateFile)
        
        self.waitingTime = self.getTotalWaitingTime()
        self.haltedVehicles = self.getTotalHaltedVehicles()
    
    def close(self):
        traci.close(False)
            
    rewardFns = {"diff_halted": diffHalted,
                "diff_waitingTime": diffWaitingTime,
                "diff_cumulativeWaitingTime": diffAccumulatedWaitingTime}