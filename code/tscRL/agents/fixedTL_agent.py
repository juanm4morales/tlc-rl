import numpy as np
import random

import sys
import os

class FixedTLAgent:
    """
        Fixed TL agent
        
        Attributes:
            environment : SumoEnvironment
                The environment where agent acts in
            currentState : tuple
                Current state
    """
    
    def __init__(self, environment, episodes, programID=None):
        environment.fixedTL = True
        self.environment = environment
        self.currentState = environment.getCurrentState()
        self.episodes = episodes
        if programID != None:
            self.environment.setTLProgram(programID)
        
    def run(self):
        metrics = []
        for episode in range(self.episodes):
            print("episode: " + str(episode+1))
            step = 0
            done = False
            cumulativeReward = 0
            meanWaitingTimeSum = 0
            while not done:
                newState, reward, done, info = self.environment.step()
                cumulativeReward = cumulativeReward + reward
                meanWaitingTimeSum += info["mean_waiting_time"]
                if done:
                    break        
                self.currentState = newState
                step += 1
                
            self.environment.reset()
            meanWaitingTime = meanWaitingTimeSum/step
            metrics.append({"episode": episode, "cumulative_reward": cumulativeReward, "mean_waiting_time": meanWaitingTime})
            
        self.environment.close()
        return metrics
        