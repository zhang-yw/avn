#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import signal
import argparse
import numpy as np
import networkx as nx
import h5py
import random

from scene_loader import THORDiscreteEnvironment
from utils.tools import SimpleImageViewer

# target = np.array([-4,2])

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
  res_li = []
  for i in range(memory['obs'].shape[0]):
    res = np.linalg.norm(memory['features'][i]-feature)
    res_li.append(res)
  top_t_li = sorted(range(len(res_li)), key=lambda i: res_li[i])[:3] 
  if res_li[top_t_li[0]] < 0.1:
    return memory['locs'][top_t_li[0]], memory['orientations'][top_t_li[0]], True
  else:
    x_s = memory['locs'][top_t_li[:]][0]
    x = int(2*(x_s[0] + x_s[1] + x_s[2])/3.0+0.5)/2.0
    y_s = memory['locs'][top_t_li[:]][1]
    y = int(2*(y_s[0] + y_s[1] + y_s[2])/3.0+0.5)/2.0
    return [x,y], memory['orientations'][top_t_li[0]], True

def next_point(graph, now_loc, target):
  now_id = node_set.index(tuple(now_loc))
  target_id = node_set.index(tuple(target))
  path = nx.astar_path(graph, source=now_id, target=target_id)
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


def navigate(env, graph, random_m):
  while not env.terminal:
    feature = env.feature
    loc, ori, _ = getSimilar(random_m, feature)
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


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-s", "--scene_dump", type=str, default="./data/bedroom_04.h5",
                      help="path to a hdf5 scene dump file")
  args = parser.parse_args()

  print("Loading scene dump {}".format(args.scene_dump))
  env = THORDiscreteEnvironment({
    'h5_file_path': args.scene_dump
  })

  random_m = h5py.File("./memory/random_walk.h5", "r")
  graph = build_graph(random_m)
  target = node_set[random.randint(0, len(node_set)-1)]
  print("random target:", target)

  env.reset()

  viewer = SimpleImageViewer()
  viewer.imshow(env.observation)

  navigate(env, graph, random_m)
