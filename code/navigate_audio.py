#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import signal
import argparse
import numpy as np
import networkx as nx
import h5py
import copy
import time

from scene_loader import THORDiscreteEnvironment
from utils.tools import SimpleImageViewer

#target = np.array([1, 2])

def build_graph(memory):
  graph = nx.Graph()
  global node_set
  node_set = []
  node_set.append(tuple(memory['locs'][0]))
  graph.add_node(len(node_set)-1)
  for i in range(memory['obs'].shape[0] - 1):
    if tuple(memory['locs'][i+1]) not in node_set:
      node_set.append(tuple(memory['locs'][i+1]))
      graph.add_node(len(node_set)-1)
    former = node_set.index(tuple(memory['locs'][i]))
    latter = node_set.index(tuple(memory['locs'][i+1]))
    graph.add_edge(former, latter)
  return graph

def getSimilar(memory, feature):
  min_f = float('inf')
  min_rank = -1
  for i in range(memory['obs'].shape[0]):
    res = np.linalg.norm(memory['features'][i]-feature)
    if res < min_f:
      min_f = res
      min_rank = i
  return memory['locs'][min_rank], memory['orientations'][min_rank], True
  '''for i in range(memory['obs'].shape[0]):
    res = np.linalg.norm(memory['features'][i]-feature)
    if res < 10:
      return memory['locs'][i], memory['orientations'][i], False'''

def next_point(graph, now_loc, target):
  now_id = node_set.index(tuple(now_loc))
  target_id = node_set.index(tuple(target))
  path = nx.shortest_path(graph, source=now_id, target=target_id)
  return node_set[path[1]]

def cal_action(now_loc, next_p, now_ori):
  assert abs(now_loc[0] - next_p[0]) + abs(now_loc[1] - next_p[1]) == 1
  action_table = [[1, 0, 3, 2],[3, 2, 1, 0],[2, 1, 0, 3],[0, 3, 2, 1]]#0 for turn right, 1 for move forward, 2 for turn left, 3 for backwards
  if now_loc[0] - next_p[0] == -1:
    return action_table[0][now_ori]
  elif now_loc[0] - next_p[0] == 1:
    return action_table[1][now_ori]
  elif now_loc[1] - next_p[1] == -1:
    return action_table[2][now_ori]
  elif now_loc[1] - next_p[1] == 1:
    return action_table[3][now_ori]
  else:
    print("NoActionError")

def get_traget_loc(random_m, loc, ori, _predict):
  predict = _predict*10
  temp = predict[0]
  predict[0] = predict[1]
  predict[1] = -temp
  print(predict)
  loc_s = np.array([float(loc[0]), float(loc[1])])
  if ori == 0:
    loc_s += [predict[0], predict[1]]
  elif ori == 1:
    loc_s += [-predict[1], predict[0]]
  elif ori == 2:
    loc_s += [-predict[0], -predict[1]]
  elif ori == 3:
    loc_s += [predict[1], -predict[0]]
  else:
    print("ori error")
    exit(0)
  min_f = float('inf')
  result = [-1, -1]
  for k in random_m['locs']:
    res = np.linalg.norm(k-loc_s)
    if res < min_f:
      min_f = res
      result = k
  print(result)
  return result

def navigate(env, graph, random_m):
  n_step = 0
  max_step = env.shortest_path_distance[0]
  while not env.terminal and n_step < 2*max_step:
    n_step += 1
    feature = env.feature
    loc, ori, _ = getSimilar(random_m, feature)
    target = get_traget_loc(random_m, loc, ori, env.predict)
    # time.sleep(0.5)
    next_p = next_point(graph, loc, target)
    action = cal_action(loc, next_p, ori)
    print(str(action)+' '+str(env.state_id)+' '+str(loc)+' '+str(next_p)+' '+str(ori))
    if action == 0:
      env.step(1)
      viewer.imshow(env.observation)
      env.step(0)
      viewer.imshow(env.observation)
    elif action == 1:
      env.step(0)
      viewer.imshow(env.observation)
    elif action == 2:
      env.step(2)
      viewer.imshow(env.observation)
      env.step(0)
      viewer.imshow(env.observation)
    elif action == 3:
      env.step(2)
      viewer.imshow(env.observation)
      env.step(2)
      viewer.imshow(env.observation)
      env.step(0)
      viewer.imshow(env.observation)
  if not env.terminal:
    print("Navigate fail!")
    return 0
  else:
    print("Navigate success!")
    return n_step/max_step

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-s", "--scene_dump", type=str, default="./data/nnew1.h5",
                      help="path to a hdf5 scene dump file")
  parser.add_argument("-m", "--memory", type=str, default="./memory/random_walk_nnew1_2000step.h5",
                      help="path to a random walk memory")
  parser.add_argument("-n", "--n_episode", type=int, default=20,
                      help="number of episode to run")
  args = parser.parse_args()

  print("Loading scene dump {}".format(args.scene_dump))
  env = THORDiscreteEnvironment({
    'h5_file_path': args.scene_dump
  })

  random_m = h5py.File(args.memory, "r")
  graph = build_graph(random_m)

  viewer = SimpleImageViewer()

  results = []
  for i in range(args.n_episode):
    env.reset()
    results.append(navigate(env, graph, random_m))

  print("Success for %s times out of %s episode"%(len(np.array(results).nonzero()), args.n_episode))

  viewer.close()
