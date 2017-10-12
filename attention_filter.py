from random import randrange, randint
from math import sqrt
import hrr
import numpy as np


bias = 1
gamma = 0.9
lrate = 0.3
td_lambda = 0.5
worldSize = 20
redGoal = randrange(0, worldSize)
greenGoal = randrange(0, worldSize)
lengthHRR = 1024
signalNumber = 3
goal = 1.0

colorDimension = 1.0
stateDimension = 0.0
candidateThreshold = 0.1

epsilon_a = 0.03
epsilon_wm = 0.03

world = np.zeros([worldSize, lengthHRR])
signals = np.zeros([signalNumber, lengthHRR])
eligibility = np.zeros(lengthHRR)
reward = np.zeros([signalNumber, lengthHRR])

weights = hrr.hrr(lengthHRR)

reward[0, redGoal] = goal
reward[1, greenGoal] = goal

for i in range(worldSize):
    world[i, :] = hrr.hrr(lengthHRR)

signals[0, :] = hrr.hrri(lengthHRR)

for i in range(1, signalNumber):
    signals[i, :] = hrr.hrr(lengthHRR)
    
for episode in range(1, 100000):

    eligibility = np.zeros(lengthHRR)

    currentLocation = randrange(0, worldSize)

    workingMemory = signals[0, :]

    currentTask = randint(1, 2)

    for timestep in range(1, 100):

        currentReward = reward[currentTask, currentLocation]
        currentState = hrr.convolve(hrr.convolve(world[currentLocation, :], signals[currentTask, :]), workingMemory)
        currentValue = np.dot(currentState, weights) + bias

        # store previous information
        previousLocation = currentLocation
        previousState = currentState
        previousWM = workingMemory
        previousTask = currentTask
        previousValue = currentValue
        eligibility = td_lambda * eligibility

        # -----------------------------------------Working Memory update process----------------------------------------------
        
        # Threshold determines possible candidates for working memory mechanism
        if stateDimension < candidateThreshold:
            memoryCandidates = np.array([signals[0, :], workingMemory, signals[currentTask, :]])
        elif colorDimension < candidateThreshold:
            memoryCandidates = np.array([signals[0, :], world[currentLocation, :], workingMemory])
        else:
            memoryCandidates = np.array([signals[0, :], world[currentLocation, :], workingMemory, signals[currentTask, :]])
        
        candidateValues = np.zeros(len(memoryCandidates))

        # Establishing the best candidate for working memory
        for i in range(len(memoryCandidates)):
            candidateState = hrr.convolve(hrr.convolve(world[currentLocation, :], signals[currentTask, :]), memoryCandidates[i, :])
            candidateValues[i] = np.dot(candidateState, weights) + bias

        bestCandidate = np.argmax(candidateValues)

        workingMemory = memoryCandidates[bestCandidate, :]
        currentState = hrr.convolve(hrr.convolve(world[currentLocation, :], signals[currentTask, :]), workingMemory)
        currentValue = np.dot(currentState, weights) + bias

        if reward[currentTask, currentLocation] == goal:
            td_error = currentValue - previousValue
            eligibility = eligibility + (previousState / sqrt(2))
            weights = weights + (lrate * eligibility * td_error)
            bias = bias + (lrate*td_error)
            break

        td_error = currentValue - previousValue
        eligibility = eligibility + (previousState / sqrt(2))
        weights = weights + (lrate * eligibility * td_error)
        bias = bias + (lrate*td_error)

        previousLocation = currentLocation
        previousState = currentState
        previousWM = workingMemory
        previousTask = currentTask
        previousValue = currentValue

        currentTask = 0

        #---------------------------------------Movement update process------------------------------------------------------

        rightLocation = currentLocation + 1
        leftLocation = currentLocation - 1

        if rightLocation == worldSize:
            rightLocation = 0

        if leftLocation == -1:
            leftLocation = worldSize - 1

        leftState = hrr.convolve(hrr.convolve(world[leftLocation, :], signals[currentTask, :]), workingMemory)
        rightState = hrr.convolve(hrr.convolve(world[rightLocation, :], signals[currentTask, :]), workingMemory)

        leftValue = np.dot(leftState, weights) + bias
        rightValue = np.dot(rightState, weights) + bias

        previousLocation = currentLocation
        previousValue = currentValue

        if leftValue <= rightValue:
            currentLocation = rightLocation
            currentValue = rightValue
        elif rightValue < leftValue:
            currentLocation = leftLocation
            currentValue = leftValue        

        currentState = hrr.convolve(hrr.convolve(world[currentLocation, :], signals[currentTask, :]), workingMemory)
        currentValue = np.dot(currentState, weights) + bias

        td_error = (currentReward + gamma * currentValue) - previousValue
        eligibility = eligibility + (previousState / sqrt(2))
        weights = weights + (lrate * eligibility * td_error)
        bias = bias + (lrate*td_error)
