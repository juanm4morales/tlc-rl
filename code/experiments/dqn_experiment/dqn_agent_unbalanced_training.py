import sys
import os

fileDir = os.path.dirname(__file__)
sys.path.append(os.path.join(fileDir, '..', '..'))

import time

import tscRL.environments.environment
from tscRL.environments.environment import SumoEnvironment, TrafficLight as tl
from tscRL.environments.rewardFn import *
from tscRL.agents.dqn_agent import DQNAgent
# Include sumo-tools directory
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

if __name__ == "__main__": 
    sumoCfgFile_unfair = os.path.abspath(os.path.join(fileDir, '../../../nets/2x2_intersection/intersection_WE.sumocfg'))
    diff_cwt_rw = DiffCWaitingTime()
    env_nfA = SumoEnvironment(
        sumocfgFile=sumoCfgFile_unfair,
        deltaTime=5,
        yellowTime=4,
        minGreenTime=10,
        gui=False,
        edges=False,
        encodeIntervals={"waitingTime":20},
        maxEncodeValue={"waitingTime":2500},
        laneInfos=["waitingTime"],
        rewardFn=diff_cwt_rw,
    )

    dqn_agent_nf = DQNAgent(
        env=env_nfA,
        learningRate=0.01,
        batchSize=64,
        explorationFraction=0.5,
        initialEpsilon = 0.8,
        verbose=1, 
    )

    dqn_agent_nf.learn(50)

    #dqn_agent.model.save("dqn_agent"+int(str(time.time())))