# -*- coding: utf-8 -*-
import sys
import h5py
import json
import numpy as np
import random
import skimage.io
from skimage.transform import resize
from constants import ACTION_SIZE
from constants import SCREEN_WIDTH
from constants import SCREEN_HEIGHT
from constants import HISTORY_LENGTH
import time

target_loc = [[-11, 1.5], [-5.5, 4], [0, -2], [-1.5, 5], [-5.5, 1], [-1.5, 2], [-1, 6]]

class THORDiscreteEnvironment(object):

  def __init__(self, config=dict()):

    # configurations
    self.scene_name          = config.get('scene_name', 'nnew1')
    self.random_start        = config.get('random_start', True)
    self.n_feat_per_locaiton = config.get('n_feat_per_locaiton', 1) # 1 for no sampling
    self.terminal_state_id   = config.get('terminal_state_id', 110)
    self.scene_rank           = config.get('scene_num', 0)
    # self.f_3 = h5py.File("/Users/yw-zhang/Desktop/cvpr_code_for227/source_predict_227.h5", "r")

    self.h5_file_path = config.get('h5_file_path', 'data/%s.h5'%self.scene_name)
    self.h5_file      = h5py.File(self.h5_file_path, 'r')

    self.locations   = self.h5_file['location'][()]
    self.rotations   = self.h5_file['rotation'][()]
    self.n_locations = self.locations.shape[0]

    self.terminals = np.zeros(self.n_locations)
    self.terminal_state_id = []

    terminal_loc = target_loc[self.scene_rank]

    for i in range(self.n_locations):
      if self.locations[i][0] >= terminal_loc[0]-0.5 and self.locations[i][0] <= terminal_loc[0]+0.5:
        if self.locations[i][1] >= terminal_loc[1]-0.5 and self.locations[i][1] <= terminal_loc[1]+0.5:
          self.terminals[i] = 1
          self.terminal_state_id.append(i)

    self.transition_graph = self.h5_file['graph'][()]
    self.shortest_path_distances = self.h5_file['shortest_path_distance'][()]

    self.history_length = HISTORY_LENGTH
    self.screen_height  = SCREEN_HEIGHT
    self.screen_width   = SCREEN_WIDTH

    # we use pre-computed fc7 features from ResNet-50
    # self.s_t = np.zeros([self.screen_height, self.screen_width, self.history_length])
    # self.s_t      = np.zeros([2048, self.history_length])
    # self.s_t1     = np.zeros_like(self.s_t)
    # self.s_target = self._tiled_state(self.terminal_state_id)

    # self.reset()

  # public methods

  def reset(self):
    # randomize initial state
    while True:
      k = random.randrange(self.n_locations)
      min_d = np.inf
      # check if target is reachable
      for t_state in self.terminal_state_id:
        dist = self.shortest_path_distances[k][t_state]
        min_d = min(min_d, dist)
      # min_d = 0  if k is a terminal state
      # min_d = -1 if no terminal state is reachable from k
      if min_d > 0: break
    self.init_state_id = k
    self.current_state_id = k

    # reset parameters
    # self.current_state_id = 300
    # self.s_t = self._tiled_state(self.current_state_id)

    self.reward   = 0
    self.collided = False
    self.terminal = False
    print("Init location:%s \t Terminal location:%s"%(self.locations[self.init_state_id], self.locations[self.terminal_state_id[0]]))

  def step(self, action):
    # time.sleep(0.5)
    assert not self.terminal, 'step() called in terminal state'
    k = self.current_state_id
    if self.transition_graph[k][action] != -1:
      self.current_state_id = self.transition_graph[k][action]
      if self.terminals[self.current_state_id]:
        self.terminal = True
        self.collided = False
      else:
        self.terminal = False
        self.collided = False
    else:
      self.terminal = False
      self.collided = True

    self.reward = self._reward(self.terminal, self.collided)
    # self.s_t1 = np.append(self.s_t[:,1:], self.state, axis=1)

  # def update(self):
  #   self.s_t = self.s_t1

  # private methods

  def _tiled_state(self, state_id):
    k = random.randrange(self.n_feat_per_locaiton)
    f = self.h5_file['resnet_feature'][state_id][:,np.newaxis]
    return np.tile(f, (1, self.history_length))

  def _reward(self, terminal, collided):
    # positive reward upon task completion
    if terminal: return 10.0
    # time penalty or collision penalty
    return -0.1 if collided else -0.01

  # properties

  @property
  def action_size(self):
    # move forward/backward, turn left/right for navigation
    return ACTION_SIZE 

  @property
  def action_definitions(self):
    action_vocab = ["MoveForward", "RotateRight", "RotateLeft", "MoveBackward"]
    return action_vocab[:ACTION_SIZE]

  @property
  def observation(self):
    return self.h5_file['observation'][self.current_state_id]

  @property
  def feature(self):
    return self.h5_file['resnet_feature'][self.current_state_id]

  @property
  def state(self):
    # read from hdf5 cache
    k = random.randrange(self.n_feat_per_locaiton)
    return self.h5_file['resnet_feature'][self.current_state_id][:,np.newaxis]

  # @property
  # def target(self):
  #   return self.s_target

  @property
  def x(self):
    return self.locations[self.current_state_id][0]

  @property
  def z(self):
    return self.locations[self.current_state_id][1]

  @property
  def r(self):
    return self.rotations[self.current_state_id]

  @property
  def state_id(self):
    return self.current_state_id

  @property
  def predict(self):
    # p_locs = np.array(self.f_3['locs'])
    # p_pres = np.array(self.f_3['predicts'])
    # for i in range(304):
    #   if abs(self.x - p_locs[i][0]) <=0.3 and abs(self.z - p_locs[i][1]) <= 0.3 and self.r == p_locs[i][2]*90:
    #     return p_pres[i]
    # print("not found")
    # exit(0)
    return self.h5_file['predict_source'][self.current_state_id]
  
  @property
  def shortest_path_distance(self):
    init_distances = self.shortest_path_distances[self.init_state_id][self.terminal_state_id]
    current_distances = self.shortest_path_distances[self.current_state_id][self.terminal_state_id]
    return min(init_distances), min(current_distances)


if __name__ == "__main__":
  scene_name = 'bedroom_04'

  env = THORDiscreteEnvironment({
    'random_start': True,
    'scene_name': scene_name,
    'h5_file_path': 'data/%s.h5'%scene_name
  })
