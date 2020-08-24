import ai2thor.controller
import h5py
import numpy as np


if __name__ == '__main__':

	f = h5py.File("./data/FloorPlan227.h5", "w")

	controller = ai2thor.controller.Controller()
	controller.local_executable_path = "/home/chenpeihao/Projects/avn/ai2thor/unity/builds/thor-local-OSXIntel64.app/Contents/MacOS/thor-local-OSXIntel64"
	controller.start()
	controller.reset('FloorPlan227_2')
	controller.step(dict(action='Initialize', gridSize=0.5))
	obs = []
	location = []
	rotation = []
	graph = []

	for i in range(520):
		obs.append(np.zeros((300, 400, 3), dtype=np.dtype('uint8')))
		location.append(np.zeros(2))
		rotation.append(-1)
		graph.append(np.zeros((4,), dtype=int))

	try:
		for X in range(13):
			for Z in range(10):
				for V in range(4):
					event = controller.step(dict(action='TeleportFull', x=-6.5+X*0.5, y=1.0, z=0.5+Z*0.5, rotation=90.0*V))
					if event.metadata['lastActionSuccess'] and event.metadata['agent']['position']['y'] == 1.0:
						location[X*40+Z*4+V][0] = -6.5+X*0.5
						location[X*40+Z*4+V][1] = 0.5+Z*0.5
						rotation[X*40+Z*4+V] = 90*V
						obs[X*40+Z*4+V] = event.frame

						flag = 0

						event = controller.step(dict(action='MoveAhead'))
						if event.metadata['lastActionSuccess']:
							newX = int((event.metadata['agent']['position']['x']+6.5)/0.5)
							newZ = int((event.metadata['agent']['position']['z']-0.5)/0.5)
							newV = int(event.metadata['agent']['rotation']['y']/90)
							assert newV == V
							graph[X*40+Z*4+V][0] = newX*40+newZ*4+newV
							event = controller.step(dict(action='MoveBack'))
						else:
							graph[X*40+Z*4+V][0] = -1

						flag = 1

						event = controller.step(dict(action='RotateRight'))
						if event.metadata['lastActionSuccess']:
							newV = int(event.metadata['agent']['rotation']['y']/90)
							assert newV == (V+1)%4
							graph[X*40+Z*4+V][1] = X*40+Z*4+newV
							event = controller.step(dict(action='RotateLeft'))
						else:
							print(str(location[X*40+Z*4+V])+" can't rotate right.")
							graph[X*40+Z*4+V][1] = -1

						flag = 2

						event = controller.step(dict(action='RotateLeft'))
						if event.metadata['lastActionSuccess']:
							newV = int(event.metadata['agent']['rotation']['y']/90)
							assert newV == (V-1)%4
							graph[X*40+Z*4+V][2] = X*40+Z*4+newV
							event = controller.step(dict(action='RotateRight'))
						else:
							print(str(location[X*40+Z*4+V])+" can't rotate left.")
							graph[X*40+Z*4+V][2] = -1

						flag = 3

						event = controller.step(dict(action='MoveBack'))
						if event.metadata['lastActionSuccess']:
							newX = int((event.metadata['agent']['position']['x']+6.5)/0.5)
							newZ = int((event.metadata['agent']['position']['z']-0.5)/0.5)
							newV = int(event.metadata['agent']['rotation']['y']/90)
							assert newV == V
							graph[X*40+Z*4+V][3] = newX*40+newZ*4+newV
							event = controller.step(dict(action='MoveAhead'))
						else:
							graph[X*40+Z*4+V][3] = -1
	except Exception as e:
		print('except:', e)
		print("X:"+str(X)+" Z:"+str(Z)+" V:"+str(V)+" flag:"+str(flag))

	f.create_dataset('graph', data = np.array(graph))
	f.create_dataset('location', data = np.array(location))
	f.create_dataset('rotation', data = np.array(rotation))
	f.create_dataset('observation', data = np.array(obs))
	f.close()