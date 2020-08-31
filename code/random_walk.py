#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import argparse
import numpy as np
import h5py
import random
import copy 

from scene_loader import THORDiscreteEnvironment

def update_loc(direction, location):
  if direction == 0:
    location[0] += 1
  elif direction == 1:
    location[1] += 1
  elif direction == 2:
    location[0] -= 1
  elif direction == 3:
    location[1] -= 1
  else:
    print('directionError')


def rollout(env, steps, save_path):

  current_loc = np.array([0,0])
  current_orientation = 1 #relative to initial orientation: 0 for right, 1 for forward, 2 for left, 3 for backwards
  features = []
  obs = []
  locs = []
  orientations = []
  state_ids = []
  f = h5py.File(save_path, "w")
  random.seed()


  for i in range(steps):
    print(i)
    features.append(env.feature)
    obs.append(env.observation)
    #print(current_loc)
    locs.append(copy.deepcopy(current_loc))
    orientations.append(current_orientation)
    state_ids.append(env.state_id)
    if random.random() > 0.5:
      random_action = random.randint(0, 2)
    else:
      random_action = 0
    env.step(random_action)
    while env.collided:
      if random.random() > 0.5:
        random_action = random.randint(0, 2)
      else:
        random_action = 0
      env.step(random_action)
    if random_action == 0:
      update_loc(current_orientation, current_loc)
    elif random_action == 1:
      current_orientation = (current_orientation - 1)%4
    elif random_action == 2:
      current_orientation = (current_orientation + 1)%4
    else:
      print(random_action)
      print('actionError')

  f.create_dataset('features', data = np.array(features))
  f.create_dataset('obs', data = np.array(obs))
  f.create_dataset('locs', data = np.array(locs))
  f.create_dataset('orientations', data = np.array(orientations))
  f.create_dataset('state_ids', data = np.array(state_ids))
  f.close()



if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("-step", "--steps", type=int, default=500,
                      help="steps of random walk")
  parser.add_argument("-scene", "--scene_dump", type=str, default="./data/FloorPlan227.h5",
                      help="path to a hdf5 scene dump file")
  parser.add_argument("-save", "--save_path", type=str, default="./memory/FloorPlan227.h5",
                      help="path to a hdf5 random walk dump file")
  args = parser.parse_args()

  print("Steps of random walk {}".format(args.steps))
  print("Loading scene dump {}".format(args.scene_dump))
  env = THORDiscreteEnvironment({
    'h5_file_path': args.scene_dump
  })
  steps = args.steps
  save_path = args.save_path

  # manually disable terminal states
  env.terminals = np.zeros_like(env.terminals)
  env.terminal_states, = np.where(env.terminals)
  env.reset()

  rollout(env, steps, save_path)
