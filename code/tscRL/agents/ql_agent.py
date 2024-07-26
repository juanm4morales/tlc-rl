import numpy as np
import random
from time import time

import sys
import os

class QLAgent:
    """
        Q Learning agent with epsilon greedy policy
        
        Attributes:
            environment : SumoEnvironment
                The environment where agent acts in
            currentState : tuple
                Current state
            lastReward : float
                Last reward aquired
            gamma : float
                Discount rate
            alpha : float
                Learning rate
            startEpsilon : float
                Exploration probability at start
            endEpsilon : float
                Minimum exploration probability
            decayRate : float
                Exponential decay rate for exploration probability
    """
    
    def __init__(self, environment, gamma, alpha, startEpsilon=1, endEpsilon=0.001, decayRate=0.02, episodes=1):
        self.environment = environment
        self.currentState = tuple(environment.getCurrentState())

        self.lastReward = 0
        # self.accReward = 0
        # self.action = None
        self.gamma = gamma
        self.alpha = alpha
        self.startEpsilon = startEpsilon
        self.endEpsilon = endEpsilon
        self.decayRate = decayRate
        self.episodes = episodes
        self.action_space = self.environment.action_space
        
        self.qTable = {self.currentState: [0 for action in range(self.action_space.n)]}
        #print(self.qTable[self.currentState][0])
        
        
    def epsilonGreedyPolicy(self, state, epsilon):
        randint = random.uniform(0,1)
        if randint > epsilon:
            action = max(self.qTable[state].items(), key=lambda x: x[1])[0]
            # action = np.argmax(self.qTable[state])
        else:
            action = int(self.action_space.sample())
        return action
    
    def deleteKnowledge(self):
        self.qTable = {
            self.currentState: {
                (action + self.action_space.start): 0 for action in range(self.action_space.n)
            }
        }
    
    def learn(self):
        metrics = []
        for episode in range(self.episodes):
            # Epsilon value with exponential decay
            epsilon = self.startEpsilon
            step = 0
            done = False
            cumulativeReward = 0
            meanWaitingTimeSum = 0
            startTime = time()
            while not done:
                action = self.epsilonGreedyPolicy(self.currentState, epsilon)
                newState, reward, _, done, info = self.environment.step(action)
                newState = tuple(newState)
                cumulativeReward = cumulativeReward + reward
                meanWaitingTimeSum += info["mean_waiting_time"]

                if newState not in self.qTable:
                    self.qTable[newState] = {(action + self.action_space.start) : 0 for action in range(self.action_space.n)}
                    
                self.qTable[self.currentState][action] = self.qTable[self.currentState][action] + self.alpha * (reward + self.gamma * max(self.qTable[newState]) - self.qTable[self.currentState][action])
                if done:
                    break
                    
                self.currentState = newState
                epsilon = self.endEpsilon + (self.startEpsilon - self.endEpsilon) * np.exp(-self.decayRate*episode)
                step += 1
                
            self.environment.reset()
            meanWaitingTime = meanWaitingTimeSum/step
            # print(cumulativeReward)
            endTime = time()
            metrics.append({"episode": episode, "cumulative_reward": cumulativeReward, "mean_waiting_time": meanWaitingTime, "elapsed_time": endTime-startTime})
    
        self.environment.close()
        return metrics
    
    def run(self, episodes=1):
        metrics = []
        for episode in range(episodes):
            # print("episode: " + str(episode+1))
            step = 0
            done = False
            cumulativeReward = 0
            meanWaitingTimeSum = 0
            while not done:
                action = max(self.qTable[state].items(), key=lambda x: x[1])[0]
                newState, reward, _, done, info = self.environment.step(action)
                
                cumulativeReward = cumulativeReward + reward
                meanWaitingTimeSum += info["mean_waiting_time"]

                if newState not in self.qTable:
                    self.qTable[newState] = {action: 0 for action in self.environment.actionSpace}
            
                self.qTable[self.currentState][action] = self.qTable[self.currentState][action] + self.alpha * (reward + self.gamma * max(self.qTable[newState].values()) - self.qTable[self.currentState][action])
                if done:
                    break
                self.currentState = newState
                step += 1
                
            self.environment.reset()
            meanWaitingTime = meanWaitingTimeSum/step
            # print("Mean Waiting Time: " + meanWaitingTime)
            metrics.append({"episode": episode, "cumulative_reward": cumulativeReward, "mean_waiting_time": meanWaitingTime})
    
        self.environment.close()                     
        return metrics
    