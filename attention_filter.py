from random import randrange, randint
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
defaultReward = 0

colorDimension = 0.0
stateDimension = 0.0
candidateThreshold = 0.0

epsilon_a = 0.03
epsilon_wm = 0.03

world = np.zeros([worldSize, lengthHRR])
signals = np.zeros([signalNumber, lengthHRR])
eligibility = np.zeros(lengthHRR)
reward = np.zeros([signalNumber, lengthHRR])

weights = hrr.hrr(lengthHRR)

reward[0, redGoal] = 1
reward[1, greenGoal] = 1

for i in range(worldSize):
    world[i,] = hrr.hrr(lengthHRR)

signals[0, ] = hrr.hrri(lengthHRR)

for i in range(1, signalNumber):
    signals[i, ] = hrr.hrr(lengthHRR)
    
for episode in range(1, 100000):

    eligibility = np.zeros(lengthHRR)

    currentLocation = randrange(0, worldSize)

    workingMemory = signals[0, ]

    currentTask = randint(1, 2)

    for timestep in range(1, 100):

        currentReward = reward[currentTask][currentLocation]
        currentState = hrr.convolve(hrr.convolve(world[currentLocation, ], signals[currentTask, ]), workingMemory)
        currentValue = np.dot(currentState, weights) + bias

        previousLocation = currentLocation
        previousState = currentState
        previousWM = workingMemory
        previousTask = currentTask
        previousValue = currentValue
        eligibility = td_lambda * eligibility

        # Working Memory Update process
        #
        # Threshold determines candidates for working memory mechanism
        
        if stateDimension < candidateThreshold:
            memoryCandidates = np.array([workingMemory, signals[currentTask, ], signals[0, ]])
        elif colorDimension < candidateThreshold:
            memoryCandidates = np.array([signals[0, ], world[currentLocation, ], workingMemory])
        else:
            memoryCandidates = np.array([signals[0, ], world[currentLocation, ], workingMemory, signals[currentTask, ]])
        