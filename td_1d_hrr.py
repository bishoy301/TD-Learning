#!/usr/bin/env python

#-------------------------------Imports------------------------------------------
import numpy as np
from random import randrange, randint
import hrr

#-------------------------------Variable Declerations----------------------------

bias = 0

gamma = 0.9         # Future Discount

lrate = 0.1

td_lambda = 0.3     # The eligibility trace lambda

worldSize = 30

goal = 19

lengthHRRs = 128

default_reward = 0

world = np.zeros([worldSize,lengthHRRs])

eligibility = np.zeros(lengthHRRs)
reward = np.zeros(worldSize)

# Setting up the weight vector
# used by the Neural Network
weights = hrr.hrr(lengthHRRs)

#-------------------------------Environment Initialization-----------------------

# Creating the environment
for i in range(worldSize):
    world[i,] = hrr.hrr(lengthHRRs)
    reward[i] = 0;


reward[goal] = 1;

#--------------------------------Learning----------------------------------------

for i in range(0, 5000):
    eligibility = np.zeros(lengthHRRs)

    # Set the starting location to a random state
    currentLocation = 0

    for j in range (0, 100):

    # Grabbing the reward for the current state r[s]
        currentReward = reward[currentLocation]

        currentValue = np.dot(world[currentLocation, ], weights) + bias

        if currentLocation == goal:
            previousLocation = currentLocation
            previousValue = currentValue
            td_error = currentReward - previousValue

            for i in range(lengthHRRs):
                (eligibility[i] * td_lambda) + world[previousLocation, ]
                weights[i] + (lrate * eligibility[i] * td_error)

            bias = bias + lrate * td_error

            break

        if currentLocation == 0:
            leftValue = np.dot(world[worldSize - 1], weights) + bias
        else:
            leftValue = np.dot(world[currentLocation - 1], weights) + bias

        if currentLocation == worldSize - 1:
            rightValue = np.dot(world[0], weights) + bias
        else:
            rightValue = np.dot(world[currentLocation + 1], weights) + bias

        previousLocation = currentLocation
        previousValue = currentValue

        if leftValue <= rightValue:
            currentLocation = currentLocation + 1
            currentValue = rightValue
        elif rightValue < leftValue:
            currentLocation = currentLocation - 1
            currentValue = leftValue

        currentReward = reward[currentLocation]

        td_error = (currentReward + gamma * currentValue) - previousValue

        for i in range(lengthHRRs):
            (eligibility[i] * td_lambda) + world[previousLocation, ]
            weights[i] + (lrate * eligibility[i] * td_error)

        bias = bias + lrate * td_error
