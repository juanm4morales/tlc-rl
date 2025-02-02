from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, List

import gymnasium as gym
from gymnasium import spaces

import traci

class IEnv(ABC):
    def __init__(self):
        self.haltedVehicles = 0
        self.cWaitingTime = 0
        self.vehWTList = []
    
    def getArrivalCWTList(self):
        pass
        
    def getTotalHaltedVehicles(self):
        pass
    
    def getTotalWaitingTime(self):
        pass
    
    def getTotalCWaitingTime(self):
        pass
    
    def getJainIndex(self):
        pass
        