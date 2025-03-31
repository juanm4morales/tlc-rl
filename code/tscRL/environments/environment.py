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


# Ensure SUMO environment variable is set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
    
# Directory for saving SUMO simulation states
state_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.states')
os.makedirs(state_dir, exist_ok=True)

# Constants for vehicle state information
HALTED = "halted"
WAITING_TIME = "waitingTime"

class Vehicle:
    """
    Represents a vehicle in the simulation with basic attributes like position, length, and max speed.
    """
    def __init__(self, position, length, maxSpeed):
        self.position = position
        self.length = length
        self.maxSpeed = maxSpeed
        self.waitingTime = 0
    
    def update(self, deltaTime):
        """ Updates the waiting time of the vehicle. """
        self.waitingTime += deltaTime
        
class Lane:
    """
    Represents a traffic lane in the simulation.

    Attributes:
        laneId (str): Unique identifier for the lane.
        lastStepHaltedVehicles (int): Number of vehicles that were halted in the last simulation step.
        lastStepWaitingTime (float): Total waiting time accumulated in the lane in the last step.
        edge (bool): Determines whether lane data should be retrieved from the edge or the lane.
    """
    def __init__(self, laneId, laneLength, edge=False):
        # self.vehicleMinGap = vehicleMinGap
        # self.vehicles = []
        self.laneId = laneId
        # self.laneLength = laneLength
        self.lastStepHaltedVehicles = 0
        self.lastStepWaitingTime = 0
        self.edge = edge
        
    def update(self):
        """ Updates traffic data for the lane using SUMO APIs. """
        if (self.edge):
            self.lastStepHaltedVehicles = traci.edge.getLastStepHaltingNumber(self.laneId)
            self.lastStepWaitingTime = traci.edge.getWaitingTime(self.laneId)
        else:
            self.lastStepHaltedVehicles = traci.lane.getLastStepHaltingNumber(self.laneId)
            self.lastStepWaitingTime = traci.lane.getWaitingTime(self.laneId)
        

class TrafficLight:
    """
    Represents a traffic light controller within the SUMO simulation.

    This class manages traffic light phases, including green, yellow, and red transitions.

    Attributes:
        id (str): The identifier for the traffic light.
        currentPhase (int): The index of the current phase.
        nextPhase (int): The index of the next phase to transition to.
        yellowTime (int): Duration for the yellow transition phase.
        minGreenTime (int): Minimum time required for a green phase before a change.
        yellow (bool): Flag to indicate if the current phase is in the yellow transition.
        currentPhaseTime (int): Counter tracking the duration of the current phase.
    """
    class Phase:
        """
        Encapsulates the state representation for a single traffic light phase.

        Attributes:
            state (str): The traffic light state string (e.g., "G" for green, "r" for red).
            yellowTransition (str): The state string used during the yellow transition.
        """
        def __init__(self, state, yellowTransition):
            self.state = state
            self.yellowTransition = yellowTransition
    # Predefined phases for the traffic light controller.
    # Each phase includes a state and a corresponding yellow transition.
    PHASES = [
        Phase("GGGgrrrrGGGgrrrr", "yyyyrrrryyyyrrrr"),  # ns_sn_l: North-South with left turn priority
        Phase("rrrrGGGgrrrrGGGg", "rrrryyyyrrrryyyy"),    # ew_we_l: East-West with left turn priority
        Phase("rrrrrrrrGGGGrrrr", "rrrrrrrryyyyrrrr"),    # sn: South-North without left turn
        Phase("GGGGrrrrrrrrrrrr", "yyyyrrrrrrrrrrrr"),    # ns: North-South without left turn
        Phase("rrrrrrrrrrrrGGGG", "rrrrrrrrrrrryyyy"),    # we: West-East without left turn
        Phase("rrrrGGGGrrrrrrrr", "rrrryyyyrrrrrrrr"),    # ew: East-West without left turn
        Phase("GGGrrrrrGGGrrrrr", "yyyrrrrryyyrrrrr"),      # ns_sn: Combined North-South phases
        Phase("rrrrGGGrrrrrGGGr", "rrrryyyrrrrryyyr"),      # ew_we: Combined East-West phases
        #Phase("rrrGrrrrrrrGrrrr", "rrryrrrrrrryrrrr"),# ne_sw
        #Phase("rrrrrrrGrrrrrrrG", "rrrrrrryrrrrrrry") # wn_es
        Phase("rrrrrrrrrrrrrrrr", "rrrrrrrrrrrrrrrr"),      # init: Initial state (all red)
    ]
    
    def __init__(self, id, initialPhase, yellowTime, minGreenTime):
        self.id = id
        # The index of the initial phase is set to the last element in the PHASES list
        self.initIndex = len(self.PHASES)-1
        self.currentPhase = self.initIndex
        self.nextPhase = self.initIndex
        self.yellowTime = yellowTime
        self.minGreenTime = minGreenTime
        self.yellow = False
        self.currentPhaseTime = 0
            
    @property
    def actionSpace(self):
        """ Returns the number of available phases as discrete actions. """
        return spaces.Discrete(len(self.PHASES)-1, start=0)
            
    def update(self):
        """
        Update the traffic light phase if a phase change is scheduled.
        Also increments the time counter for the current phase.
        """
        if (self.currentPhase != self.nextPhase):
            if (self.yellow and self.currentPhaseTime >= self.yellowTime) or not self.yellow:
        # Transition to the next phase by setting the appropriate state string.

                nextPhaseState = self.PHASES[self.nextPhase].state
                traci.trafficlight.setRedYellowGreenState(self.id, nextPhaseState)
                self.currentPhase = self.nextPhase
                self.currentPhaseTime = 0
                self.yellow = False
        # Progress the simulation and update phase time.
        traci.simulationStep() 
        self.currentPhaseTime += 1
    
    def canChange(self):
        """
        Check if the traffic light is allowed to change phase based on elapsed time.

        Returns:
            bool: True if the current phase has lasted longer than both the minimum green and yellow times.
        """
        return (
            not (self.yellow)
            and (self.currentPhaseTime >= self.minGreenTime)
            and (self.currentPhaseTime >= self.yellowTime)
            )
    
    def changePhase(self, newPhase):
        """
        Request a phase change to the specified new phase.

        If the current phase is different from the new phase and the conditions are met (or if initializing),
        then the method sets the yellow transition and schedules the phase change.

        Args:
            newPhase (int): The index of the new phase to transition to.

        Returns:
            int: Returns the previous phase time if the transition was scheduled,
                 -1 if the minimum phase duration has not been met,
                 -2 if the new phase is the same as the current phase.
        """
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
    """
    Represents the state of the simulation, including the traffic light phase and discretized lane information.

    Attributes:
        tlPhase (int): Current traffic light phase index.
        discreteLaneInfo (list): A list of discretized values representing lane metrics.
        discreteClass (Discrete): An instance used to convert raw lane metrics into discrete values.
    """
    def __init__(self, tlPhase, lanes: Dict[str, Lane], discreteClass, laneInfo="halted"):
        self.discreteClass = discreteClass
        self.tlPhase = tlPhase
        # Discretize lane metrics based on the selected type ('halted' or 'waitingTime')
        self.discreteLaneInfo = self.discretizeLaneInfo(lanes, laneInfo)

    def getTupleState(self):
        """
        Return the state as a tuple including the traffic light phase and lane information.
        """
        return (self.tlPhase, *self.discreteLaneInfo)
    
    def getArrayState(self):
        """
        Return the state as a NumPy array by concatenating the traffic light phase and lane info.
        """
        return np.append(self.tlPhase, self.discreteLaneInfo)

    def discretizeLaneInfo(self, lanes: Dict[str, Lane], laneInfo):
        """
        Convert raw lane metrics into discrete values using a logarithmic interval.

        Args:
            lanes (dict): Dictionary of Lane objects.
            laneInfo (str): Specifies which metric to use ('waitingTime' or 'halted').

        Returns:
            list: A list of discretized values for each lane.
        """
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
    """
    Farama Gym-compatible environment for traffic signal control using SUMO.

    This environment encapsulates the SUMO simulation, defines action and observation spaces,
    and provides functions to step through the simulation and reset it.

    Attributes:
        simTime (int): Total simulation time.
        deltaTime (int): Time step interval for the simulation.
        fixedTL (bool): Flag to determine if the traffic light operates under a fixed program.
        lanes (dict): Dictionary of Lane objects controlled by the traffic light.
        rewardFn (function): Function used to compute the reward for an action.
    """
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
        """
        Get the current simulation time step.
        """
        return traci.simulation.getTime()
    
    @property
    def actionSpace(self):
        """
        List the available phases (actions) for the traffic light.
        """
        return [actionKey for actionKey in self.trafficLight.PHASES if actionKey != 'init']
       
    def _initializeSimulation(self):
        """
        Starts the SUMO simulation with the specified configuration and parameters.
        """
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
        """
        Runs the simulation for a given warming time to stabilize initial conditions.
        
        Args:
            warmingTime (int): Number of simulation steps to warm up.
        """
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
        """
        Sets the traffic light program based on a given ID.

        Args:
            programID (int): The identifier for the desired traffic light program.
        """
        try:
            traci.trafficlight.setProgram(tls_ids[0], programID)
        except traci.TraCIException as traci_e:
            print(traci_e, end="")
            print(" Program ID setted to 1.")
            traci.trafficlight.setProgram(tls_ids[0], "1")
            
    def getCurrentState(self):
        """
        Retrieve the current state of the environment.

        Returns:
            np.array: A numerical representation combining traffic light phase and discretized lane metrics.
        """
        state = State(self.trafficLight.currentPhase, self.lanes, self.discreteClass, self.laneInfo)
        return state.getArrayState()
        #return state.getTupleState()
    
    def _getTotalHaltedVehicles(self):
        """
        Calculate the total number of halted vehicles across all lanes.
        """
        return sum(lane.lastStepHaltedVehicles for lane in self.lanes.values())
    
    def _getTotalWaitingTime(self):
        """
        Calculate the total waiting time across all lanes.
        """
        return sum(lane.lastStepWaitingTime for lane in self.lanes.values())
    
    def computeReward(self):
        """
        Compute the reward for the current simulation step using the selected reward function.
        
        Returns:
            float: The computed reward.
        """
        return self.rewardFn(self)
    
    def _getAccumulatedWaitingTime(self):
        """
        Compute the accumulated waiting time for all vehicles currently in the simulation.
        
        Returns:
            float: Total accumulated waiting time.
        """
        accumulatedWaitingTime = 0
        for lane in self.lanes.values():
            vehicles = traci.lane.getLastStepVehicleIDs(lane.laneId)
            for vehicle in vehicles:
                accumulatedWaitingTime += traci.vehicle.getAccumulatedWaitingTime(vehicle)
                
        return accumulatedWaitingTime
    
    def _diffHalted(self):
        """
        Compute reward as the difference in the number of halted vehicles between steps.
        """
        currentStepHaltedVehicles = self._getTotalHaltedVehicles()
        reward = self.haltedVehicles-currentStepHaltedVehicles
        self.haltedVehicles = currentStepHaltedVehicles
        return reward

    def _diffWaitingTime(self):
        """
        Compute reward as the difference in waiting time between steps.
        """
        currentWaitingTime = self._getTotalWaitingTime()
        reward = self.waitingTime - currentWaitingTime
        self.waitingTime = currentWaitingTime
        return reward

    def _diffAccumulatedWaitingTime(self):
        """
        Compute reward as the difference in accumulated waiting time between steps.
        """
        currentAccWaitingTime = self._getAccumulatedWaitingTime()
        reward = self.cumulativeWaitingTime - currentAccWaitingTime
        self.cumulativeWaitingTime = currentAccWaitingTime
        return reward

    def getInfo(self):
        """
        Retrieve additional information from the simulation, such as average waiting time.

        Returns:
            dict: A dictionary containing simulation step, mean waiting time, and mean accumulated waiting time.
        """        
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
        """
        Advance the simulation by one step.

        Args:
            action (int, optional): The action to apply (i.e., new phase for the traffic light).

        Returns:
            tuple: A tuple containing the new state, reward, done flag (always False), 
                   truncated flag (if simulation ended), and additional info.
        """
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
        
        # Retrieve new state, compute reward, and check termination conditions.
        state = self.getCurrentState()
        reward = self.computeReward()
        truncated = traci.simulation.getMinExpectedNumber() == 0 or traci.simulation.getTime() > self.simTime
  
        info = self.getInfo()
        return state, reward, False, truncated, info
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.

        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Additional options for resetting.

        Returns:
            tuple: The initial state and additional info.
        """
        super().reset(seed=seed)
        # Reset traffic light parameters to initial conditions.
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
            # Load the saved state from the warm-up phase.
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
        """
        Close the SUMO simulation connection.
        """
        traci.close()
        

    def __del__(self):
        """
        Destructor to ensure the simulation is closed upon deletion of the environment.
        """
        self.close()
        
    # Define reward function mappings (functions defined later in the class)     
    rewardFns = {"diff_halted": _diffHalted,
                "diff_waitingTime": _diffWaitingTime,
                "diff_cumulativeWaitingTime": _diffAccumulatedWaitingTime}