import sys
import os
import traci
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
    
sumoCMD = ["sumo-gui", "-c", "nets/2x2_intersection/intersection_balanced.sumocfg"]

traci.start(sumoCMD)
tls_ids = traci.trafficlight.getIDList()
lanesIds = traci.trafficlight.getControlledLanes(tls_ids[0])
lanes = []
for laneId in lanesIds:
    if "in" in laneId:
        if laneId not in lanes:
            lanes.append(laneId)
traci.trafficlight.setProgram(tls_ids[0], "1")

while True:
    traci.simulationStep()
    accumulatedWaitingTime = 0
    waitingTime = 0
    for laneId in lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(laneId)
            waitingTime += traci.lane.getWaitingTime(laneId) 
            for vehicle in vehicles:
                accumulatedWaitingTime += traci.vehicle.getAccumulatedWaitingTime(vehicle)
    print(
        
    )
    print("--------------------------------------")            
    print("Acc Waiting Time: " + str(accumulatedWaitingTime))
    print("--------------------------------------")
    print("Waiting Time: " + str(waitingTime))
    print("--------------------------------------")
