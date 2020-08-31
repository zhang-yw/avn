import h5py
import os
import networkx as nx
import numpy as np

def id2node(id):
    return id // 4

def build_graph(transition):
    graph = nx.Graph()
    for idx, trans in enumerate(transition):
        if id2node(idx) not in graph.nodes:
            graph.add_node(id2node(idx))
        for latter in trans:
            if latter != -1:
                graph.add_edge(id2node(idx), id2node(latter))
    return graph

if __name__ == "__main__":
    for scene_name in os.listdir("./data/environment/"):
        if not scene_name.endswith(".h5"):
            continue
        print(scene_name)
        in_path = "./data/environment/%s" % scene_name
        h5_file_in = h5py.File(in_path, 'r+')
        transidtion = h5_file_in["graph"][()]
        location = h5_file_in["location"]

        graph = build_graph(transidtion)
        n_location = len(location)
        shortest_path_distances = np.zeros((n_location, n_location))
        for s_idx in range(n_location):
            for e_idx in range(n_location):
                if s_idx == e_idx:
                    shortest_path_distance = 0
                else:
                    try:
                        shortest_path_distance = nx.shortest_path_length(graph, source=id2node(s_idx), target=id2node(e_idx))
                    except nx.exception.NetworkXNoPath:
                        shortest_path_distance = np.inf
                shortest_path_distances[s_idx, e_idx] = shortest_path_distance


        if "shortest_path_distance" in h5_file_in.keys():
            h5_file_in.__delitem__("shortest_path_distance")
        if "shortest_path_distances" in h5_file_in.keys():
            h5_file_in.__delitem__("shortest_path_distances")
        h5_file_in.create_dataset("shortest_path_distance", data=shortest_path_distances)
        h5_file_in.close()
