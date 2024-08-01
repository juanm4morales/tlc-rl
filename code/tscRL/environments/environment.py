import sys
import os
import random
import numpy as np
from typing import Dict

import gymnasium as gym
from gymnasium import spaces

import traci
from sumolib import checkBinary

from tscRL.util.discrete import Discrete

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
    PHASES = [  Phase("GGGgrrrrGGGgrrrr", "yyyyrrrryyyyrrrr"), # ns_sn_l
                Phase("rrrrGGGgrrrrGGGg", "rrrryyyyrrrryyyy"), # ew_we_l
                Phase("rrrrrrrrGGGGrrrr", "rrrrrrrryyyyrrrr"), # sn
                Phase("GGGGrrrrrrrrrrrr", "yyyyrrrrrrrrrrrr"), # ns
                Phase("rrrrrrrrrrrrGGGG", "rrrrrrrrrrrryyyy"), # we
                Phase("rrrrGGGGrrrrrrrr", "rrrryyyyrrrrrrrr"), # ew
                Phase("GGGrrrrrGGGrrrrr", "yyyrrrrryyyrrrrr"), # ns_sn
                Phase("rrrrGGGrrrrrGGGr", "rrrryyyrrrrryyyr"), # ew_we
                #Phase("rrrGrrrrrrrGrrrr", "rrryrrrrrrryrrrr"),# ne_sw
                #Phase("rrrrrrrGrrrrrrrG", "rrrrrrryrrrrrrry") # wn_es
                Phase("rrrrrrrrrrrrrrrr", "rrrrrrrrrrrrrrrr"), # init
    ]
    
    def __init__(self, id, initialPhase, yellowTime, minGreenTime):
        self.id = id
        self.initIndex = len(self.PHASES)-1
        self.currentPhase = self.initIndex
        self.nextPhase = self.initIndex
        self.yellowTime = yellowTime
        self.minGreenTime = minGreenTime
        self.yellow = False
        self.currentPhaseTime = 0
            
    @property
    def actionSpace(self):
        return spaces.Discrete(len(self.PHASES)-1, start=0)
            
    def update(self):
        if (self.currentPhase != self.nextPhase):
            if (self.yellow and self.currentPhaseTime >= self.yellowTime) or not self.yellow:
                nextPhaseState = self.PHASES[self.nextPhase].state
                traci.trafficlight.setRedYellowGreenState(self.id, nextPhaseState)
                self.currentPhase = self.nextPhase
                self.currentPhaseTime = 0
                self.yellow = False
                
        traci.simulationStep() 
        self.currentPhaseTime += 1
    
    def canChange(self):
        return (
            not (self.yellow)
            and (self.currentPhaseTime >= self.minGreenTime)
            and (self.currentPhaseTime >= self.yellowTime)
            )
    
    def changePhase(self, newPhase):
        if (self.currentPhase != newPhase):
            if self.canChange() or (self.currentPhase == self.initIndex):
                if (self.currentPhase != self.initIndex):  
                    yellowPhaseState = self.PHASES[self.currentPhase].yellowTransition
                    traci.trafficlight.setRedYellowGreenState(self.id, yellowPhaseState)
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
    
    def getArrayState(self):
        return np.append(self.tlPhase, self.discreteLaneInfo)

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
    MAX_VEH_LANE = 30      # adjust according to lane length? Param?
    MAX_WAITING_TIME = 500 # Param?
    def __init__(
        self,
        sumocfgFile,
        deltaTime=5,
        yellowTime=4,
        minGreenTime=5,
        gui=False,
        edges=False,
        discreteIntervals=6,
        maxLaneValue=60,
        laneInfo="halted",
        rewardFn="diff_halted",
        fixedTL=False,
        simTime=43800,
        warmingTime=600,
        sumoLog=False,
        waitingTimeMemory=1000
    ) -> None:
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
        
        self.laneInfo = laneInfo
        if rewardFn in self.rewardFns.keys():
            self.rewardFn = self.rewardFns[rewardFn]
        else:
            self.rewardFn = self.rewardFns["diff_halted"]
            print("Warning: Invalid rewardFn value. \"diff_halted\" value was assigned instead.")
        
        self.sumoLog = sumoLog
        self.waitingTimeMemory = waitingTimeMemory
        
        # Start SUMO, load network, set waiting time memory
        self._initializeSimulation()
        
        tls_ids = traci.trafficlight.getIDList()
        
        self.trafficLight = TrafficLight(tls_ids[0], 0, yellowTime, minGreenTime)  
        lanesIds = traci.trafficlight.getControlledLanes(tls_ids[0])
        self.lanes: Dict[str, Lane] = {}
        for laneId in lanesIds:
            if "in" in laneId:
                if edges:
                    laneId = traci.lane.getEdgeID(laneId) 
                if laneId not in self.lanes:
                    self.lanes[laneId] = Lane(laneId, traci.lane.getLength(laneId), edge = edges)
        
        # Discrete Class. For encoding lane info
        self.discreteClass = Discrete(discreteIntervals, maxLaneValue)
        warmingTime = 600
        #Warming up
        self._warmingUpSimulation(warmingTime)
        
        self.totalTimeSteps = self.simTime // self.deltaTime
        
        # Program ID
        if self.fixedTL:
            traci.trafficlight.setProgram(tls_ids[0], "2")
        else:
            traci.trafficlight.setProgram(tls_ids[0], "0")
            
        # Action space
        self.action_space = self.trafficLight.actionSpace

        # Observation space
        low = np.zeros(len(self.lanes)+1)
        high = np.full(self.action_space.n + 1, discreteIntervals)
        high[0] = self.action_space.n
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int64)
        

    @property
    def simStep(self):
        return traci.simulation.getTime()
    
    @property
    def actionSpace(self):
        return [actionKey for actionKey in self.trafficLight.PHASES if actionKey != 'init']
       
    def _initializeSimulation(self):
        sumoCMD = [self.sumoBinary, "-c", self.sumocfgFile, "--waiting-time-memory", str(self.waitingTimeMemory)
                   #"--tripinfo-output", "tripinfo.xml"
                   ]
        
        if not(self.sumoLog):
            sumoCMD.append("--no-step-log")
            sumoCMD.append("--no-warnings")
        if self.gui:
            sumoCMD.append("-S")
            sumoCMD.append("--quit-on-end")
        
        try:
            traci.start(sumoCMD)
            
        except traci.TraCIException as traci_e:
            print(traci_e, end="")
            print(" Closing connection...")
            traci.close()
            from time import sleep
            sleep(1)
            print(traci_e, end="")
            print(" Starting new connection...")
            traci.start(sumoCMD)
            print("OK")
    
    def _warmingUpSimulation(self, warmingTime):
        traci.simulationStep(warmingTime-1) # Warming Time
        traci.trafficlight.setRedYellowGreenState(self.trafficLight.id, self.trafficLight.PHASES[0].state)
        traci.simulationStep()
        self.waitingTime = self._getTotalWaitingTime()
        self.haltedVehicles = self._getTotalHaltedVehicles()
        self.trafficLight.currentPhase = self.trafficLight.initIndex
        self.trafficLight.nextPhase = self.trafficLight.initIndex
        for lane in self.lanes.values():
            lane.update()
        traci.simulation.saveState(self.stateFile)
        
    def _setTLProgram(self, programID: int):
        try:
            traci.trafficlight.setProgram(tls_ids[0], programID)
        except traci.TraCIException as traci_e:
            print(traci_e, end="")
            print(" Program ID setted to 1.")
            traci.trafficlight.setProgram(tls_ids[0], "1")
            
    def getCurrentState(self):
        state = State(self.trafficLight.currentPhase, self.lanes, self.discreteClass, self.laneInfo)
        return state.getArrayState()
        #return state.getTupleState()
    
    def _getTotalHaltedVehicles(self):
        return sum(lane.lastStepHaltedVehicles for lane in self.lanes.values())
    
    def _getTotalWaitingTime(self):
        return sum(lane.lastStepWaitingTime for lane in self.lanes.values())
    
    def computeReward(self):
        return self.rewardFn(self)
    
    def _getAccumulatedWaitingTime(self):
        accumulatedWaitingTime = 0
        for lane in self.lanes.values():
            vehicles = traci.lane.getLastStepVehicleIDs(lane.laneId)
            for vehicle in vehicles:
                accumulatedWaitingTime += traci.vehicle.getAccumulatedWaitingTime(vehicle)
                
        return accumulatedWaitingTime
    
    def _diffHalted(self):
        currentStepHaltedVehicles = self._getTotalHaltedVehicles()
        reward = self.haltedVehicles-currentStepHaltedVehicles
        self.haltedVehicles = currentStepHaltedVehicles
        return reward

    def _diffWaitingTime(self):
        currentWaitingTime = self._getTotalWaitingTime()
        reward = self.waitingTime - currentWaitingTime
        self.waitingTime = currentWaitingTime
        return reward

    def _diffAccumulatedWaitingTime(self):
        currentAccWaitingTime = self._getAccumulatedWaitingTime()
        reward = self.cumulativeWaitingTime - currentAccWaitingTime
        self.cumulativeWaitingTime = currentAccWaitingTime
        return reward

    def getInfo(self):        
        vehicleCount = traci.vehicle.getIDCount()
        
        if (self.rewardFn in [self.rewardFns["diff_halted"], self.rewardFns["diff_cumulativeWaitingTime"]]):
            self.waitingTime = self._getTotalWaitingTime()
        elif (self.rewardFn == self.rewardFns["diff_waitingTime"]):
            self.haltedVehicles = self._getTotalHaltedVehicles()
            
        meanWaitingTime = self.waitingTime / vehicleCount if vehicleCount > 0 else 0
        
        if self.rewardFn != self.rewardFns["diff_cumulativeWaitingTime"]:
            self.cumulativeWaitingTime = self._getAccumulatedWaitingTime()
            
        meanAccWaitingTime = self.cumulativeWaitingTime / vehicleCount if vehicleCount > 0 else 0

        info = {
            "sim_step": self.simStep,
            "mean_waiting_time": meanWaitingTime,
            "mean_acc_waiting_time": meanAccWaitingTime
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
        reward = self.computeReward()
        truncated = traci.simulation.getMinExpectedNumber() == 0 or traci.simulation.getTime() > self.simTime
  
        info = self.getInfo()
        return state, reward, False, truncated, info
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.trafficLight.yellow = False
        self.trafficLight.currentPhase = self.trafficLight.initIndex
        self.trafficLight.nextPhase = self.trafficLight.initIndex
        self.trafficLight.currentPhaseTime = 0

        self.haltedVehicles = 0
        self.waitingTime = 0
        self.cumulativeWaitingTime = 0
        
        for laneKey in self.lanes:
            self.lanes[laneKey].lastStepHaltedVehicles = 0
            self.lanes[laneKey].waitingTime = 0
            
        try:
            traci.simulation.loadState(self.stateFile)
        except traci.TraCIException:
            self._initializeSimulation()
            traci.simulation.loadState(self.stateFile)
        
        self.waitingTime = self._getTotalWaitingTime()
        self.haltedVehicles = self._getTotalHaltedVehicles()
        
        state = self.getCurrentState()
        info = self.getInfo()
        
        return state, info
    
    def close(self):
        traci.close()
        

    def __del__(self):
        self.close()
            
    rewardFns = {"diff_halted": _diffHalted,
                "diff_waitingTime": _diffWaitingTime,
                "diff_cumulativeWaitingTime": _diffAccumulatedWaitingTime}