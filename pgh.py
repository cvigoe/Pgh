from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import random
import pandas as pd

from collections import Counter
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa


def run():
    
    # os.system('cls')
    
    graph = Graph()
    LANES = list(traci.lane.getIDList())

    for lane in LANES:
        allowed = traci.lane.getAllowed(lane)
        if allowed == [] or 'private' in allowed:
            continue
        else:
            LANES.remove(lane)

    edges = []
    EDGE_to_LANE = {}

    for lane in LANES:
        conseq = traci.lane.getLinks(lane)
        allowed_conseq = []
        for conseq_lane in conseq:
            allowed = traci.lane.getAllowed(conseq_lane[0])
            if allowed == [] or 'private' in allowed:
                allowed_conseq.append(conseq_lane[0])
        print(cost)
        cost = traci.lane.getTraveltime(lane)

        for conseq_lane in allowed_conseq:
            edges.append( (lane, conseq_lane, cost) )

        EDGE_to_LANE[traci.lane.getEdgeID(lane)] = lane

    for edge in edges:
        graph.add_edge(*edge)

    start_times = Counter({})
    end_times = Counter({})
    distances = Counter({})
    vehicle_record = {}
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        t = traci.simulation.getTime()
        print('Time: ', t)

        VEHICLES = traci.vehicle.getIDList()

        for vehicle in VEHICLES:
            if vehicle not in start_times.keys():
                start_times[vehicle] = t

                origin = traci.vehicle.getLaneID(vehicle)
                old_route = traci.vehicle.getRoute(vehicle)
                destination = EDGE_to_LANE[old_route[-1]]
                new_route = dijsktra(graph, origin, destination)
                if new_route == 'Route Not Possible':
                    # print('\n', vehicle, origin, 'Route not possible', '\n', sep='\n')
                    continue
                new_route = [traci.lane.getEdgeID(i) for i in new_route]
                if traci.lane.getEdgeID(origin) in old_route:
                    old_route = old_route[old_route.index(traci.lane.getEdgeID(origin)):]
                    if len(set(old_route) - set(new_route)) > 0:
                        print('\n', vehicle, origin, set(old_route) - set(new_route), '\n', sep='\n')
                    traci.vehicle.setRoute(vehicle,new_route)
            end_times[vehicle] = t
            distances[vehicle] = traci.vehicle.getDistance(vehicle)


        # global_weights.update_weights()

    travel_times = end_times - start_times
    normed_travel_times = Counter({key:travel_times[key]/distances[key] for key in travel_times.keys()})

    plt.hist(normed_travel_times.values(),bins=20)
    plt.title('Normed Travel Times')
    plt.show()
    plt.hist(travel_times.values(),bins=20)
    plt.title('Travel Times')
    plt.show()


def dijsktra(graph, initial, end):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()
    
    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)
        
        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
    
    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

class Graph_Weights:
    def __init__(self, edges):
        self.edges = edges
        self.weights = dict.fromkeys(edges,1)

    def update_weights(self):
        for edge in self.edges:
            self.weights[edge] = traci.edge.getTraveltime(edge)

class Graph():
    def __init__(self):
        """
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights has all the weights between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """
        self.edges = defaultdict(list)
        self.weights = {}
    
    def add_edge(self, from_node, to_node, weight):
        # Note: assumes edges are bi-directional
        self.edges[from_node].append(to_node)
        self.weights[(from_node, to_node)] = weight


# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", "pgh.sumocfg","--tripinfo-output", "pgh.xml", "--no-step-log", "--gui-settings-file", "pgh_style.xml", "--tls.actuated.show-detectors", "TRUE"])
    run()
