import sys
import os


fileDir = os.path.dirname(__file__)
sys.path.append(os.path.join(fileDir, '..', '..'))

import time

import tscRL.environments.environment
from tscRL.environments.environment import SumoEnvironment, TrafficLight as tl

from tscRL.agents.dqn_agent import DQNAgent
# Include sumo-tools directory
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

if __name__ == "__main__": 
    sumoCfgFile_unbalanced = os.path.abspath(os.path.join(fileDir, '../../../nets/2x2_intersection/intersection_unbalanced.sumocfg'))

    env = SumoEnvironment(
        sumocfgFile=sumoCfgFile_unbalanced,
        deltaTime=5,
        yellowTime=4,
        minGreenTime=10,
        gui=False,
        edges=False,
        discreteIntervals=20,
        maxLaneValue=2500, 
        laneInfo="waitingTime",
        rewardFn="diff_cumulativeWaitingTime",
        fixedTL=False,
        simTime=43800, 
        sumoLog=False
    )

    dqn_agent = DQNAgent(
        env=env,
        learningRate=0.001,
        batchSize=64,
        explorationFraction=0.5,
        verbose=1)

    dqn_agent.learn(100)
    dqn_agent.model.save("dqn_agent"+int(str(time.time())))