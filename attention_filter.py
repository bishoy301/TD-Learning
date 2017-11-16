from random import randrange, randint, random
from math import sqrt, floor
import matplotlib.pyplot as plt
import hrr
import numpy as np


bias = 1
gamma = 0.9
lrate = 0.1
td_lambda = 0.3
worldSize = 20
redGoal = 0 #randrange(0, worldSize)
greenGoal = floor(worldSize / 2) #randrange(0, worldSize)
lengthHRR = 1024
signalNumber = 4
goal = 1.0

isWM = True
dimensions = []
dimensions.append(1.0)
dimensions.append(0.0)
update_dim = None

candidateThreshold = 0.1

epsilon_a = 0.03
epsilon_wm = 0.03

world = np.zeros([worldSize, lengthHRR])
signals = np.zeros([signalNumber, lengthHRR])
memory = np.zeros([signalNumber, lengthHRR])
eligibility = np.zeros(lengthHRR)
reward = np.zeros([signalNumber, worldSize])

weights = hrr.hrr(lengthHRR, normalized=True)

reward[1, redGoal] = goal
reward[2, greenGoal] = goal

for i in range(worldSize):
    world[i, :] = hrr.hrr(lengthHRR, normalized=True)

signals[0, :] = hrr.hrri(lengthHRR)
memory[0, :] = hrr.hrri(lengthHRR)

for i in range(1, signalNumber):
    signals[i, :] = hrr.hrr(lengthHRR, normalized=True)
    memory[i, :] = hrr.hrr(lengthHRR, normalized=True)


def getState(location, task, mem):
    State = hrr.convolve(hrr.convolve(world[location, :], signals[task, :]), memory[mem, :])
    return State

def getValue(location, state, mem):
    if mem:
        Value = np.dot(state, weights) + bias
        return Value

    Value = np.dot(world[location,:], weights) + bias
    return Value

for episode in range(1, 10000):

    print("This is a new episode. Number {}".format(episode))
    print("  ")
    print("  ")
    print("  ")
    print("  ")
    print("  ")
    print("  ")
    print("  ")
    print("  ")

    eligibility = np.zeros(lengthHRR)

    currentLocation = randrange(0, worldSize)

    workingMemory = 0;

    currentTask = randint(1, 2)
    currentSignal = currentTask

    for timestep in range(1, 100):

        if dimensions[0] == 0.0 and dimensions[1] == 0.0:
            isWM = False

        currentReward = reward[currentSignal, currentLocation]
        #currentState = hrr.convolve(hrr.convolve(world[currentLocation, :], signals[currentTask, :]), memory[workingMemory, :])
        currentState = getState(currentLocation, currentTask, workingMemory)
        #currentValue = np.dot(currentState, weights) + bias
        currentValue = getValue(currentLocation, currentState, isWM)

        # store previous information
        previousLocation = currentLocation
        previousState = currentState
        previousWM = workingMemory
        previousTask = currentTask
        previousValue = currentValue
        eligibility = td_lambda * eligibility

        # -----------------------------------------Working Memory update process----------------------------------------------

        # Threshold determines possible candidates for working memory mechanism
        if dimensions[1] < candidateThreshold:
            memoryCandidates = np.array([signals[0, :], memory[workingMemory, :], signals[currentTask, :]])
            mem = 1
        elif dimensions[0] < candidateThreshold:
            memoryCandidates = np.array([signals[0, :], world[currentLocation, :], memory[workingMemory, :]])
            mem = 2
        else:
            memoryCandidates = np.array([signals[0, :], world[currentLocation, :], memory[workingMemory, :], signals[currentTask, :]])
            mem = 3

        candidateValues = []

        # Establishing the best candidate for working memory
        for row in range(memoryCandidates.shape[0]):
            candidateState = hrr.convolve(hrr.convolve(world[currentLocation, :], signals[currentTask, :]), memoryCandidates[row, :])
            candidateValues.append(np.dot(candidateState, weights) + bias)

        bestCandidate = np.argmax(candidateValues)

        if random() < epsilon_wm:
            workingMemory = randint(0, 2)
            print("Epsilon! Working Memory")
        else:
            if mem == 1:
                if bestCandidate == 0:
                    workingMemory = 0
                elif bestCandidate == 1:
                    if workingMemory == 1 or workingMemory == 2:
                        update_dim = 0
                    elif workingMemory == 3:
                        update_dim = 1
                    workingMemory = workingMemory
                elif bestCandidate == 2:
                    workingMemory = currentTask
            elif mem == 2:
                    if bestCandidate == 0:
                        workingMemory = 0
                    elif bestCandidate == 1:
                        workingMemory = 3
                    elif bestCandidate == 2:
                        if workingMemory == 1 or workingMemory == 2:
                            update_dim = 0
                        elif workingMemory == 3:
                            update_dim = 1
                        workingMemory = workingMemory
            elif mem == 3:
                    if bestCandidate == 0:
                        workingMemory = 0
                    elif bestCandidate == 1:
                        workingMemory = 3
                    elif bestCandidate == 2:
                        if workingMemory == 1 or workingMemory == 2:
                            update_dim = 0
                        elif workingMemory == 3:
                            update_dim = 1
                        workingMemory = workingMemory
                    elif bestCandidate == 3:
                        workingMemory = currentTask

        print(candidateValues)

        print(bestCandidate)

        #currentState = hrr.convolve(hrr.convolve(world[currentLocation, :], signals[currentTask, :]), memory[workingMemory, :])
        currentState = getState(currentLocation, currentTask, workingMemory)
        #currentValue = np.dot(currentState, weights) + bias
        currentValue = getValue(currentLocation, currentState, isWM)

        if reward[currentSignal, currentLocation] == goal:
            td_error = currentValue - previousValue
            eligibility = eligibility + (previousState)
            weights = weights + (lrate * eligibility * td_error)
            if update_dim is not None:
                dimensions[update_dim] = dimensions[update_dim] * td_error
            #bias = bias + (lrate*td_error)
            print("Reward at current location is {}".format(reward[currentSignal, currentLocation]))

            print("Found Goal! At state {} with a value of {}".format(currentLocation, currentValue))
            print("The td_error is {}".format(td_error))

            td_error = currentReward - currentValue
            eligibility = eligibility + (previousState)
            weights = weights + (lrate * eligibility * td_error)
            #bias = bias + (lrate*td_error)
            break

        td_error = currentValue - previousValue
        eligibility = eligibility + (previousState)
        weights = weights + (lrate * eligibility * td_error)
        #bias = bias + (lrate*td_error)

        previousLocation = currentLocation
        previousState = currentState
        previousWM = workingMemory
        previousTask = currentTask
        previousValue = currentValue

        print("The current value of state {} during task {} with current signal of {} and a working memory of {} is {}".format(currentLocation, currentSignal, currentTask, bestCandidate, currentValue))

        currentTask = 0

        #---------------------------------------Movement update process------------------------------------------------------

        rightLocation = currentLocation + 1
        leftLocation = currentLocation - 1

        if rightLocation == worldSize:
            rightLocation = 0

        if leftLocation == -1:
            leftLocation = worldSize - 1

        #leftState = hrr.convolve(hrr.convolve(world[leftLocation, :], signals[currentTask, :]), memory[workingMemory, :])
        leftState = getState(leftLocation, currentTask, workingMemory)
        #rightState = hrr.convolve(hrr.convolve(world[rightLocation, :], signals[currentTask, :]), memory[workingMemory, :])
        rightState = getState(rightLocation, currentTask, workingMemory)

        #leftValue = np.dot(leftState, weights) + bias
        leftValue = getValue(leftLocation, leftState, isWM)
        #rightValue = np.dot(rightState, weights) + bias
        rightValue = getValue(rightLocation, rightState, isWM)

        previousLocation = currentLocation
        previousValue = currentValue

        print("Right is state {} with a value of {}".format(rightLocation, rightValue))
        print("Left is state {} with a value of {}".format(leftLocation, leftValue))

        if leftValue <= rightValue:
            if random() < epsilon_a:
                currentLocation = leftLocation
                print("Epsilon! Moving.")
            else:
                currentLocation = rightLocation
        elif rightValue < leftValue:
            if random() < epsilon_a:
                currentLocation = rightLocation
                print("Epsilon! Moving")
            else:
                currentLocation = leftLocation

        #currentState = hrr.convolve(hrr.convolve(world[currentLocation, :], signals[currentTask, :]), memory[workingMemory, :])
        currentState = getState(currentLocation, currentTask, workingMemory)
        #currentValue = np.dot(currentState, weights) + bias
        currentValue = getValue(currentLocation, currentState, isWM)

        td_error = (currentReward + gamma * currentValue) - previousValue
        eligibility = eligibility + (previousState)
        weights = weights + (lrate * eligibility * td_error)
        #bias = bias + (lrate*td_error)
        print("The td_error is {}".format(td_error))


# Plotting the data

s0_w0 = []
s0_w1 = []
s0_w2 = []

s1_w0 = []
s1_w1 = []
s1_w2 = []

s2_w0 = []
s2_w1 = []
s2_w2 = []

for i in range(worldSize):
    state0 = hrr.convolve(hrr.convolve(world[i, :], signals[0, :]), memory[0, :])
    s0_w0.append(np.dot(state0, weights) + bias)

    state1 = hrr.convolve(hrr.convolve(world[i, :], signals[1, :]), memory[0, :])
    s0_w1.append(np.dot(state1, weights) + bias)

    state2 = hrr.convolve(hrr.convolve(world[i, :], signals[2, :]), memory[0, :])
    s0_w2.append(np.dot(state2, weights) + bias)

    state3 = hrr.convolve(hrr.convolve(world[i, :], signals[0, :]), memory[1, :])
    s1_w0.append(np.dot(state3, weights) + bias)

    state4 = hrr.convolve(hrr.convolve(world[i, :], signals[1, :]), memory[1, :])
    s1_w1.append(np.dot(state4, weights) + bias)

    state5 = hrr.convolve(hrr.convolve(world[i, :], signals[2, :]), memory[1, :])
    s1_w2.append(np.dot(state5, weights) + bias)

    state6 = hrr.convolve(hrr.convolve(world[i, :], signals[0, :]), memory[2, :])
    s2_w0.append(np.dot(state6, weights) + bias)

    state7 = hrr.convolve(hrr.convolve(world[i, :], signals[1, :]), memory[2, :])
    s2_w1.append(np.dot(state7, weights) + bias)

    state8 = hrr.convolve(hrr.convolve(world[i, :], signals[2, :]), memory[2, :])
    s2_w2.append(np.dot(state8, weights) + bias)

plt.plot(s0_w0, 'r--', s0_w1, 'b--', s0_w2, 'g--', s1_w0, 'c--', s1_w1, 'm--', s1_w2, 'y--', s2_w0, 'k--', s2_w1, 'r-', s2_w2, "b-")
#plt.axis(0, 19, 0, 1)
plt.ylabel('Value')
plt.xlabel('States')
plt.show()


