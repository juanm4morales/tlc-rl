import sys
import os

current_dir=os.path.dirname(__file__)

sys.path.append(os.path.join(current_dir, '..'))
print(sys.path)


from tscRL.environments.environment import SumoEnvironment
from tscRL.agents.ql_agent import QLAgent

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
    
sumoCfgFile_unbalanced = os.path.abspath(os.path.join(current_dir, '../../nets/2x2_intersection/intersection_unbalanced.sumocfg'))

env = SumoEnvironment(sumocfgFile=sumoCfgFile_unbalanced, deltaTime=5, yellowTime=4, minGreenTime=10, gui=False, edges=False, discreteIntervals=4, maxLaneValue=500,  laneInfo="waitingTime", rewardFn="diff_cumulativeWaitingTime", fixedTL=False, simTime=43800, sumoLog=False)
agent = QLAgent(environment=env, gamma=0.99, alpha=0.01, startEpsilon=1, endEpsilon=0.005, decayRate=0.025, episodes=200)
qla_cWT_metrics = agent.train();



