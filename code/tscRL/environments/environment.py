import sys
import os
import random
import numpy as np
from typing import Dict, Any, Optional, List

import gymnasium as gym
from gymnasium import spaces

import traci
from sumolib import checkBinary

from tscRL.environments.rewardFn import *
from tscRL.environments.ienv import IEnv
from tscRL.util.discrete import Discrete

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
    

state_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.states')
os.makedirs(state_dir, exist_ok=True)
'''
class Vehicle:
    def __init__(self, position, length, maxSpeed):
        self.position = position
        self.length = length
        self.maxSpeed = maxSpeed
        self.waitingTime = 0
    
    def update(self, deltaTime):
        self.waitingTime += deltaTime
'''

class Lane:
    HALTED = "halted"
    WAITING_TIME = "waitingTime"
    C_WAITING_TIME ="cWaitingTime"
    def __init__(self, laneId, laneLength, laneInfos, edge=False):
        # self.vehicleMinGap = vehicleMinGap
        # self.vehicles = []
        self.laneId = laneId
        # self.laneLength = laneLength
        self.haltedVehicles = 0
        self.waitingTime = 0
        self.cWaitingTime = 0
        self.meanSpeed = 0
        self.edge = edge
        self.laneInfos = laneInfos
        for laneInfo in laneInfos:
            if (laneInfo not in self.info.keys()):
                raise ValueError("Invalid Lane Info Value. Use 'halted', 'waitingTime' or 'meanSpeed'.")
        self.vehiclesCWTList = []
        
    def _updateHaltedVehicles(self):
        if (self.edge):
            self.haltedVehicles = traci.edge.getLastStepHaltingNumber(self.laneId)
        else:
            self.haltedVehicles = traci.lane.getLastStepHaltingNumber(self.laneId)
            
    def _updateWaitingTime(self):
        if (self.edge):
            self.waitingTime = traci.edge.getWaitingTime(self.laneId)
        else:
            self.waitingTime = traci.lane.getWaitingTime(self.laneId)
         
    def _updateCWaitingTime(self):
        '''
        if (self.edge):
            self.waitingTime = traci.edge.getWaitingTime(self.laneId)
        else:
            self.waitingTime = traci.lane.getWaitingTime(self.laneId)
        '''    
        laneCWaitingTime = 0
        vehicleCWaitingTime = 0
        self.vehiclesCWTList = []
        vehicles = traci.lane.getLastStepVehicleIDs(self.laneId)
        
        for vehicle in vehicles:
            vehicleCWaitingTime = traci.vehicle.getAccumulatedWaitingTime(vehicle)
            laneCWaitingTime += vehicleCWaitingTime
            self.vehiclesCWTList.append(vehicleCWaitingTime)
        self.cWaitingTime = laneCWaitingTime
        return laneCWaitingTime
                        
    def update(self):
        for laneInfo in self.laneInfos:
            self.info[laneInfo](self)
            
        self._updateCWaitingTime()
        ## TEMPORAL!!! Solo para testeoo de WT    
        
            
    info = {HALTED: _updateHaltedVehicles,
            WAITING_TIME: _updateWaitingTime,
            C_WAITING_TIME: _updateCWaitingTime}

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
    def __init__(self, tlPhase, lanes: Dict[str, Lane], discreteClasses, laneInfos):
        self.discreteClasses = discreteClasses
        self.tlPhase = tlPhase
        self.discreteLaneInfo = self.discretizeLaneInfo(lanes, laneInfos)

    def getTupleState(self):
        return (self.tlPhase, *self.discreteLaneInfo)
    
    def getArrayState(self):
        return np.append(self.tlPhase, self.discreteLaneInfo)

    def discretizeLaneInfo(self, lanes: Dict[str, Lane], laneInfos):
        # discreteLaneQueue: Dict[str, int] = {}
        discreteLaneInfo = []
        
        if Lane.C_WAITING_TIME in laneInfos:
            for lane in lanes.values():
                discreteLaneInfo.append(self.discreteClasses[Lane.C_WAITING_TIME].log_interval(lane.cWaitingTime)) 
        
        if Lane.WAITING_TIME in laneInfos:
            for lane in lanes.values():
                discreteLaneInfo.append(self.discreteClasses[Lane.WAITING_TIME].log_interval(lane.waitingTime)) 
        if Lane.HALTED in laneInfos:
            for lane in lanes.values():
                discreteLaneInfo.append(self.discreteClasses[Lane.HALTED].log_interval(lane.haltedVehicles))

        return discreteLaneInfo          

class SumoEnvironment(IEnv, gym.Env):
    #MAX_VEH_LANE = 30      # adjust according to lane length? Param?
    #MAX_WAITING_TIME = 500 # Param?
    def __init__(
        self,
        sumocfgFile:str,
        deltaTime:int=5,
        yellowTime:int=4,
        minGreenTime:int=10,
        gui:bool=False,
        edges:bool=False,
        encodeIntervals:Dict[str, int]={Lane.WAITING_TIME:20},
        maxEncodeValue:Dict[str, int]={Lane.WAITING_TIME:2500},
        laneInfos:List[str]=[Lane.WAITING_TIME],
        rewardFn:RewardFn = DiffCWaitingTime(1.0),
        fixedTL:bool=False,
        simTime:int=43800,
        warmingTime:int=600,
        sumoLog:bool=False,
        waitingTimeMemory:int=1000
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
        assert all(laneInfo in encodeIntervals for laneInfo in laneInfos)
        assert all(laneInfo in maxEncodeValue for laneInfo in laneInfos)
        # Assert con laneInfos. QUe los valores de laneInfos esten en  las keys de encodeIntervals y maxEncodeValue
        self.deltaTime = deltaTime
        self.fixedTL=fixedTL
        self.haltedVehicles = 0
        self.cWaitingTime = 0
        self.vehWTList = []
        
        self.laneInfos = laneInfos
      
        self.rewardFn = rewardFn
        self.rewardFn.setEnv(self)
            
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
                    self.lanes[laneId] = Lane(laneId, traci.lane.getLength(laneId), laneInfos=self.laneInfos, edge = edges)
        
        # Discrete Classes. For encoding lane info
        self.discreteClasses = {}
        if Lane.WAITING_TIME in self.laneInfos:
            self.discreteClasses[Lane.WAITING_TIME] = Discrete(encodeIntervals[Lane.WAITING_TIME], maxEncodeValue[Lane.WAITING_TIME])
        if Lane.C_WAITING_TIME in self.laneInfos:
            self.discreteClasses[Lane.C_WAITING_TIME] = Discrete(encodeIntervals[Lane.C_WAITING_TIME], maxEncodeValue[Lane.C_WAITING_TIME])
        if Lane.HALTED in self.laneInfos:
            self.discreteClasses[Lane.HALTED] = Discrete(encodeIntervals[Lane.HALTED], maxEncodeValue[Lane.HALTED])
            
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
        low = np.zeros(len(self.lanes) * len(self.laneInfos) + 1)
        high = np.array([self.action_space.n])
        for maxValue in encodeIntervals.values():
            high = np.concatenate((high, np.full(len(self.lanes), maxValue)))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int64)
        self.jainIndex = 1

    @property
    def simStep(self):
        return traci.simulation.getTime()
    
    @property
    def actionSpace(self):
        return [actionKey for actionKey in self.trafficLight.PHASES if actionKey != 'init']
       
    def _initializeSimulation(self):
        sumoCMD = [self.sumoBinary,
                   "-c", self.sumocfgFile,
                   "--waiting-time-memory", str(self.waitingTimeMemory),
                   "--keep-after-arrival", str(self.deltaTime)
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
        
        self.updateLanesInfo()
        self.cWaitingTime = self.getTotalCWaitingTime()
        self.haltedVehicles = self.getTotalHaltedVehicles()
        self.trafficLight.currentPhase = self.trafficLight.initIndex
        self.trafficLight.nextPhase = self.trafficLight.initIndex
    
        traci.simulation.saveState(self.stateFile)
        
    def _setTLProgram(self, programID: int):
        try:
            traci.trafficlight.setProgram(tls_ids[0], programID)
        except traci.TraCIException as traci_e:
            print(traci_e, end="")
            print(" Program ID setted to 1.")
            traci.trafficlight.setProgram(tls_ids[0], "1")
            
    def getCurrentState(self):
        state = State(self.trafficLight.currentPhase, self.lanes, self.discreteClasses, self.laneInfos)
        return state.getArrayState()
        #return state.getTupleState()
        
    def getArrivalCWTList(self):
        arrivalCWT = []
        arrivedVehList = traci.simulation.getArrivedIDList()
        for vehId in arrivedVehList:
            arrivalCWT.append(traci.vehicle.getAccumulatedWaitingTime(vehId))
        return arrivalCWT
        
    def getTotalHaltedVehicles(self):
        return sum(lane.haltedVehicles for lane in self.lanes.values())
    
    def getTotalWaitingTime(self):
        return sum(lane.waitingTime for lane in self.lanes.values())
    
    def getTotalCWaitingTime(self):
        return sum(lane.cWaitingTime for lane in self.lanes.values())
    
    def getTotalExpCWaitingTime(self):
        return sum(sum(pow(cwt,2) for cwt in lane.vehiclesCWTList) for lane in self.lanes.values())
    
    def getJainIndex(self):
        allVehiclesWTList = []
        wTSum = 0
        for lane in self.lanes.values():
            allVehiclesWTList += lane.vehiclesWTList
            wTSum += lane.cWaitingTime
            
        self.vehWTList = allVehiclesWTList
        squaredWTSum = 0
        for vehWT in allVehiclesWTList:
            squaredWTSum += pow(vehWT, 2)
        if (squaredWTSum != 0):
            jainIndex = pow(wTSum,2)/(len(self.vehWTList)*squaredWTSum)
        else:
            jainIndex = 0

        return jainIndex
    
    def computeReward(self):
        return self.rewardFn.computeReward()
        '''
        r1 = self.mainRewardFn(self)
        r2 = 0
        weight = 0
        if self.fairRewardFn != None:
            weight = self.fairRewardWeight
            r2 = self.fairRewardFn(self)
        return r1 + weight*r2
        '''
        
    '''
    def _diffHalted(self):
        currentStepHaltedVehicles = self.getTotalHaltedVehicles()
        reward = self.haltedVehicles-currentStepHaltedVehicles
        self.haltedVehicles = currentStepHaltedVehicles
        return reward

    def _diffWaitingTime(self):
        currentWaitingTime = self.getTotalWaitingTime()
        reward = self.waitingTime - currentWaitingTime
        self.waitingTime = currentWaitingTime
        return reward

    def _diffCWaitingTime(self):
        currentCWaitingTime = self.getTotalCWaitingTime()
        reward = self.cWaitingTime - currentCWaitingTime
        self.cWaitingTime = currentCWaitingTime
        return reward

    def _jainFairness(self, negative=True):
        allVehiclesWTList = []
        wTSum = 0
        for lane in self.lanes.values():
            allVehiclesWTList += lane.vehiclesWTList
            wTSum += lane.cWaitingTime
            
        self.vehWTList = allVehiclesWTList
        squaredWTSum = 0
        for vehWT in allVehiclesWTList:
            squaredWTSum += pow(vehWT, 2)
        if (squaredWTSum != 0):
            jainIndex = pow(wTSum,2)/(len(self.vehWTList)*squaredWTSum)
        else:
            jainIndex = 0
        
        if negative:
            return -(1-jainIndex)*self.fairRewardWeight
        else:
            return jainIndex*self.fairRewardWeight
        
    def _diffJainIndex(self):
        allVehiclesWTList = []
        wTSum = 0
        for lane in self.lanes.values():
            allVehiclesWTList += lane.vehiclesWTList
            wTSum += lane.cWaitingTime
            
        self.vehWTList = allVehiclesWTList
        squaredWTSum = 0
        for vehWT in allVehiclesWTList:
            squaredWTSum += pow(vehWT, 2)
        if (squaredWTSum != 0):
            jainIndex = pow(wTSum,2)/(len(self.vehWTList)*squaredWTSum)
        else:
            jainIndex = 0
        reward = -(self.jainIndex-jainIndex)
        self.jainIndex = jainIndex
        return reward
    
    '''
    def getInfo(self):        
        #vehicleCount = traci.vehicle.getIDCount()

        #meanCWaitingTime = self.cWaitingTime / vehicleCount if vehicleCount > 0 else 0
        info = {
            "sim_step": self.simStep,
            #"mean_acc_waiting_time": meanCWaitingTime,
            "arrival_acc_waiting_times": self.getArrivalCWTList()
        }
        return info
    
    def updateLanesInfo(self):
        for lane in self.lanes.values():
            lane.update()
            
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
            
        self.updateLanesInfo()
        
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
        self.cWaitingTime = 0
        self.vehWTList = []
        self.jainIndex = 1
        
        for laneKey in self.lanes:
            self.lanes[laneKey].haltedVehicles = 0
            self.lanes[laneKey].waitingTime = 0
            self.lanes[laneKey].cWaitingTime = 0
            self.lanes[laneKey].meanSpeed = 0
            self.lanes[laneKey].vehiclesWTList = []
            
        try:
            traci.simulation.loadState(self.stateFile)
        except traci.TraCIException:
            self._initializeSimulation()
            traci.simulation.loadState(self.stateFile)
            
        self.updateLanesInfo()
        
        self.cWaitingTime = self.getTotalCWaitingTime()
        self.haltedVehicles = self.getTotalHaltedVehicles()
        
        state = self.getCurrentState()
        info = self.getInfo()
        
        return state, info
    
    def close(self):
        traci.close()
        

    def __del__(self):
        self.close()
            
    '''
    rewardFns = {DIFF_HALTED: _diffHalted,
                 DIFF_CWAITING_TIME: _diffCWaitingTime,
                }
    fairRewardFns = {JAIN_FAIRNESS: _jainFairness,
                     DIFF_N_JAIN: _diffJainIndex
                }
    '''