import h5py
import numpy as np
import networkx as nx

def get_pres(locs, rot, p_locs, p_pres):
	for i in range(len(p_locs)):
		if abs(p_locs[i][0]-locs[0]) < 0.25 and abs(p_locs[i][1]-locs[1]) < 0.25 and abs(p_locs[i][2]*90-rot) < 0.25:
			return p_pres[i]
	print("Not find")
	print(locs)
	print(rot)
	exit(0)


f = h5py.File("/Users/yw-zhang/Desktop/cvpr_code_for227/data/FloorPlan227.h5", "r")
f_2 = h5py.File("/Users/yw-zhang/Desktop/cvpr_code_for227/data/FloorPlan227_new.h5", "w")
f_3 = h5py.File("/Users/yw-zhang/Desktop/cvpr_code_for227/source_predict_227.h5", "r")
graph = []
location = []
observation = []
resnet_feature = []
rotation = []
fx = -1*np.ones(520, dtype=np.int8)
count = 0
for i in range(len(list(f['rotation']))):
	if f['rotation'][i] != -1:
		fx[i] = count
		count += 1
		graph.append(f['graph'][i])
		location.append(f['location'][i])
		observation.append(f['observation'][i])
		resnet_feature.append(f['resnet_feature'][i])
		rotation.append(f['rotation'][i])
for i in range(len(graph)):
	for j in range(4):
		if graph[i][j] != -1:
			graph[i][j] = fx[graph[i][j]]
print(np.asarray(graph).shape)
f_2.create_dataset("graph", data=np.array(graph))
f_2.create_dataset("location", data=np.array(location))
f_2.create_dataset("observation", data=np.array(observation))
f_2.create_dataset("resnet_feature", data=np.array(resnet_feature))
f_2.create_dataset("rotation", data=np.array(rotation))
g = nx.Graph()
path = -1*np.ones((304,304))
for i in range(304):
	g.add_node(i)
for i in range(304):
	for j in range(4):
		if(graph[i][j] != -1):
			g.add_edge(i, graph[i][j])
paths = nx.shortest_path(g)
print(g.nodes())
for i in paths.keys():
	for j in paths[i].keys():
		path[i][j] = len(paths[i][j])
f_2.create_dataset("shortest_path_distance", data=np.array(path))

predict = []
p_locs = np.array(f_3['locs'])
p_pres = np.array(f_3['predicts'])
for i in range(304):
	pres = get_pres(location[i], rotation[i], p_locs, p_pres)
	predict.append(pres)
f_2.create_dataset("predict_source", data=np.array(predict))

f.close()
f_2.close()
f_3.close()
