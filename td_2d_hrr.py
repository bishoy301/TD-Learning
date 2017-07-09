#!/usr/bin/env python

from random import randrange
import numpy as np
import hrr

bias = 1
gamma = 0.9
lrate = 0.1
td_lambda = 0.3
lengthHRRs = 128
default_reward = 0.0
reward = np.zeros(lengthHRRs)
worldSize = 30
world = np.zeros(lengthHRRs)
weights = hrr.hrr(lengthHRRs)
currentLocation = randrange(0, worldSize)

def Reward(state):
    currentReward = reward[state]
    return currentReward

def Value(currentLocation):
    current_value = np.dot(world[currentLocation, ], weights) + bias
    return current_value

