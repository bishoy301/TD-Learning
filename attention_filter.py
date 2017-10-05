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

world       = np.zeros([worldSize, lengthHRR])
signals     = np.zeros([signalNumber, lengthHRR])
eligibility = np.zeros(lengthHRR)
reward      = np.zeros(lengthHRR)

weights = hrr.hrr(lengthHRR)

