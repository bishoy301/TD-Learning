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
memoryCandidates = np.zeros(4)

world       = np.zeros([worldSize, lengthHRR])
signals     = np.zeros([signalNumber, lengthHRR])
eligibility = np.zeros(lengthHRR)
reward      = np.zeros([signalNumber, lengthHRR])

weights = hrr.hrr(lengthHRR)

reward[0][redGoal] = 1
reward[1][greenGoal] = 1

for i in range(worldSize):
    world[i,] = hrr.hrr(lengthHRR)
    
print(world)

