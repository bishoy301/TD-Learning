#!/usr/bin/env python

#-------------------------------Imports------------------------------------------
import numpy as np
from random import randrange, randint
import hrr

#-------------------------------Variable Declerations----------------------------

bias = 1

gamma = 0.9         # Future Discount

lrate = 0.1

td_lambda = 0.3     # The eligibility trace lambda

worldSize = 30

goal = randrange(0, worldSize)

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
    currentLocation = randrange(0, worldSize)

    for j in range (0, 100):

    # Grabbing the reward for the current state r[s]
        currentReward = reward[currentLocation]

        currentValue = np.dot(world[currentLocation, ], weights) + bias

        if currentLocation == goal:
            previousLocation = currentLocation
            previousValue = currentValue
            td_error = currentReward - previousValue


            eligibility = (eligibility * td_lambda) + world[previousLocation, ]
            weights = weights + (lrate * eligibility * td_error)

            # bias = bias + lrate * td_error

            break

        rightLocation = currentLocation + 1
        leftLocation = currentLocation - 1

        if rightLocation == worldSize:
            rightLocation = 0

        if leftLocation == -1:
            leftLocation = worldSize - 1

        leftValue = np.dot(world[leftLocation, ], weights) + bias
        rightValue = np.dot(world[rightLocation, ], weights) + bias

        previousLocation = currentLocation
        previousValue = currentValue

        if leftValue <= rightValue:
            currentLocation = rightLocation
            currentValue = rightValue
        elif rightValue < leftValue:
            currentLocation = leftLocation
            currentValue = leftValue

        td_error = (currentReward + gamma * currentValue) - previousValue

        eligibility = (eligibility * td_lambda) + world[previousLocation, ]
        weights = weights + (lrate * eligibility * td_error)

        # bias = bias + lrate * td_error

print(np.apply_along_axis(np.dot, 1, world, weights) + bias)
