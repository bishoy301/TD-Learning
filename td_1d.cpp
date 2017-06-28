/* ========================================================================
   $File: td_1d.cpp
   $Date: $
   $Revision: $
   $Creator: Bishoy Boktor $
   $Notice: (C) Copyright 2017. All Rights Reserved. $
   ======================================================================== */

#include <State.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <vector>

using namespace std;

const double GAMMA = 0.9;
int episodes = 3000;
ofstream logFile;
vector<State> environment;
int environmentSize = 10;
int currentLocation;
double td_error;
double lambda = 0.1;
double alpha = 0.1;

enum Move {
    Left,
    Right
};

double V(int);
double r(int);

int getLeft(int);
int getRight(int);
Move nextMove(State, State, State);
Move randMove();

void resetEligibilities();
void updateEligibilities(State& currentState);

int main() {


  logFile.open("log.txt");
    // initializing the environment
    for (int i = 0; i < environmentSize; i++) {
        State newState(0, 0, i);

        environment.push_back(newState);
    }

    int goal = rand() % 10;

    //setting the reward for the goal
    environment[goal].setReward(1.0);

    // Main loop
    for (int i = 0; i < episodes; i++) {
        currentLocation = rand() % environmentSize;

        do {

          if (currentLocation == goal) {

            updateEligibilities(environment[currentLocation]);
            td_error = r(currentLocation) - V(currentLocation);

            for (State& state : environment) {
              state.setValue(state.getValue() + alpha * td_error * state.getEligibility());
            }

            break;
          }

          State* previousState = &environment[currentLocation];

           int prevLocation = previousState->getIndex();

          Move movement = nextMove(environment[getLeft(currentLocation)],
                                   environment[currentLocation],
                                   environment[getRight(currentLocation)]);

          switch (movement) {
          case Left:
            currentLocation = getLeft(currentLocation);
            break;
          case Right:
            currentLocation = getRight(currentLocation);
            break;
          default:
            int randPick = rand() % 2;
            if (randPick == 0) {
              currentLocation = getLeft(currentLocation);
            } else {
              currentLocation = getRight(currentLocation);
            }
            break;
          }

          updateEligibilities(*previousState);

          td_error = (r(prevLocation) + GAMMA * V(currentLocation)) - V(prevLocation);

          for (State& state : environment) {
            state.setValue(state.getValue() + alpha * td_error * state.getEligibility());
          }

        } while (currentLocation != goal);

    }

    for (int index = 0; index < environmentSize; index++) {
      logFile << "\tState " << index << "\n\tValue: " << environment[index].getValue() << "\n\t\tReward: " << environment[index].getReward() << "\n";
      cout << "\tState " << index << "\n\tValue: " << environment[index].getValue() << "\n\t\tReward: " << environment[index].getReward() << "\n";

    }

    logFile.close();

    return 0;
}

Move nextMove(State left, State current, State right) {
    Move movement;

    if (left.getValue() < right.getValue()) {
      movement = Right;
    } else if (right.getValue() < left.getValue()) {
      movement = Left;
    } else {
      randMove();
    }
    return movement;
}

Move randMove() {
  int randPick = rand() % 2;
  Move movement;

  if (randPick == 0) {
    movement = Left;
  } else {
    movement = Right;
  }
  return movement;
}

int getLeft(int currLocation) {
    int left;

    if (currLocation == 0) {
        left = environmentSize - 1;
    } else {
        left = currLocation - 1;
    }

    return left;
}

int getRight(int currLocation) {
    int right;

    if (currLocation == environmentSize - 1) {
        right = 0;
    } else {
        right = currLocation + 1; 
    }

    return right;
}

void updateEligibilities(State& currentState) {
  for (State& state : environment) {
    state.updateEligibility(lambda);
  }

  currentState.setEligibility();
}

void resetEligibilities() {
  for (State& state : environment) {
    state.resetEligibility();
  }
}

double V(int state) {
  return environment[state].getValue();
}

double r(int state) {
  return environment[state].getReward();
}
