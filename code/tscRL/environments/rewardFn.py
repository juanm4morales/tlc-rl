from abc import ABC, abstractmethod
import traci
from tscRL.environments.ienv import IEnv

class RewardFn(IEnv, ABC):
    def __init__(self, weight:float = 1.0):
        self.env : IEnv = None
        self.weight = weight
    
    def setEnv(self, env):
        self.env = env
        
    @abstractmethod
    def computeReward(self):
        pass
class DiffHalted(RewardFn):
    def computeReward(self):
        currentHaltedVehicles = self.env.getTotalHaltedVehicles()
        reward = self.env.haltedVehicles - currentHaltedVehicles
        self.env.haltedVehicles = currentHaltedVehicles
        return reward * self.weight

class DiffCWaitingTime(RewardFn):
    def computeReward(self):
        currentCWaitingTime = self.env.getTotalCWaitingTime()
        reward = self.env.cWaitingTime - currentCWaitingTime
        self.env.cWaitingTime = currentCWaitingTime
        return reward * self.weight

class DiffExpCWaitingTime(RewardFn):
    def computeReward(self):
        currentCWaitingTime = self.env.getTotalCWaitingTime()
        reward = self.env.cWaitingTime - currentCWaitingTime
        self.env.cWaitingTime = currentCWaitingTime
        return reward * self.weight
    
class DiffNJainIndex(RewardFn):
    def computeReward(self):
        currentJainIndex =  self.env.getJainIndex()
        reward = -(self.env.jainIndex-currentJainIndex)
        self.env.jainIndex = currentJainIndex
        return reward * self.weight
    
class MORewardFn(RewardFn):
    def __init__(self, mainRewardFn:RewardFn, fairRewardFn:RewardFn):
        super().__init__()
        self.mainRewardFn = mainRewardFn
        self.fairRewardFn = fairRewardFn
    
    def setEnv(self, env):
        self.mainRewardFn.setEnv(env)
        self.fairRewardFn.setEnv(env)
    
    def computeReward(self):
        return self.mainRewardFn.computeReward() + self.fairRewardFn.computeReward()