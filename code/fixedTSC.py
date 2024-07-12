#!/usr/bin/env python

import os
import sys
import optparse


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import traci

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

def run():
    
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        vehicles_stopped = 0
        vehicles = traci.vehicle.getIDList()
        # Iterar sobre cada vehículo
        for vehicle_id in vehicles:
            # Obtener la velocidad del vehículo
            velocity = traci.vehicle.getSpeed(vehicle_id)
            # Filtrar vehículos detenidos
            if velocity == 0:
                vehicles_stopped += 1
        
        print("Step " + str(step) + ":")
        print("Vehicles stopped: " + str(vehicles_stopped))
        print("---------------------------")
                
        step += 1
    traci.close()
    sys.stdout.flush()

if __name__ == "__main__":
    options = get_options()

    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    traci.start([sumoBinary, "-c", "./nets/interseccion.sumocfg",
                             "--tripinfo-output", "tripinfo.xml"])
    run()