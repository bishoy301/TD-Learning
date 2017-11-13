/* Created by Bishoy Boktor
   Working Memory implementation and HRR Proof of Concept
   Dependencies: fftw3 and DFTTool
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "DFTTool.h"

using namespace std;
using namespace WMtk;

typedef vector<double> HRR;

const int WORLD_SIZE     = 20;
const int NUMBER_SIGNALS = 3;


enum Move {
    Left,
    Right
};

enum WM {
	current_working_memory,
	current_sig,
    nothing
};

int hrr_size = 1024;
DFTTool tool(hrr_size);

HRR getState(HRR world, HRR signal, HRR memory);
double getValue(HRR state, HRR& weight);
Move nextMove(HRR left, HRR current, HRR right, HRR current_signal, HRR current_memory, HRR weight);
Move randMove();
int getLeft(int current_location);
int getRight(int current_location);
WM memoryUpdate(HRR current_memory, HRR current_signal, HRR nothing_state, HRR current_location_HRR, HRR weight);

int main() {
    int number_episodes      = 1000;
    int steps                = 100;

    double default_reward    = 0;
    double goal_reward       = 1;
    int rand_goal1           = 0;
    int rand_goal2           = 10;
    double previous_reward;

    int bias                 = 0;
    double gamma             = 0.9;
    double learn_rate        = 0.3;
    double lambda            = 0.5;
    double td_error;

    vector<HRR> world(WORLD_SIZE);
    vector<HRR> signals(NUMBER_SIGNALS);   // The signals present in environment
    vector<HRR> memory(NUMBER_SIGNALS);    // The signals remembered by the agent
    HRR weight(hrr_size);
    HRR eligibility(hrr_size);
    int reward[NUMBER_SIGNALS][WORLD_SIZE];


    for (int i = 0; i < WORLD_SIZE; ++i) {
        world[i].push_back(tool.hrr());
    }

    signals[0].push_back(tool.ihrr());
    memory[0].push_back(tool.ihrr());
    for (int i = 1; i <= NUMBER_SIGNALS; ++i) {
        signals[i].push_back(tool.hrr());
        memory[i].push_back(tool.hrr());
    }

    for (int i = 0; i < NUMBER_SIGNALS; ++i) {
        for (int j = 0; j < WORLD_SIZE; ++j) {
            reward[i][j] = default_reward;
        }
    }

    // Choosing random goals
    reward[1][rand_goal1] = goal_reward;
    reward[2][rand_goal2] = goal_reward;

    for (int episode = 1; episode <= number_episodes; ++episode) {
        /* for (int i = 0; i < hrr_size; ++i) {
            eligibiltiy[i].push_back(0);
            } */
        std::fill(eligibility.begin(), eligibility.end(), 0);

        int current_task               = (rand() > RAND_MAX/2) ? 1 : 2; 
        int current_location           = rand() % (WORLD_SIZE);
        HRR current_location_HRR       = world[current_location];
        HRR current_memory             = memory[0];
        HRR current_signal             = signals[current_task];

        for (int timestep = 0; timestep < steps; ++timestep) {
            int current_reward         = reward[current_task][current_location];
            HRR current_state          = getState(current_location_HRR, current_signal, current_memory);
            double current_value       = getValue(current_state, weight) + bias;

            int previous_location      = current_location;
            HRR previous_state         = current_state;
            HRR previous_memory        = current_memory;
            HRR previous_signal        = current_signal;
            double previous_value      = current_value;

            for (int i = 0; i < eligibility.size(); ++i) {
                eligibility[i] *= lambda;
            }

            HRR nothing_state = memory[0];

            // Update of Working Memory
            WM memory_update = memoryUpdate(current_memory, current_signal, nothing_state, current_location_HRR, weight);

            // Making the actual memory update
            switch(memory_update) {
                case current_working_memory:
                    break;                // In the current_memory case we don't need to do anything 
				case current_sig:
                    current_memory = current_signal;
                    break;
                case nothing:
                    current_memory = nothing_state;
                    break;
                default:
                    // make default current_working_memory
                    break;
            }

            current_state = getState(current_location_HRR, current_signal, current_memory);
            current_value = getValue(current_state, weight) + bias;


            if (reward[current_task][current_location] == goal_reward) {
                td_error = current_value - previous_value;

                for (int i = 0; i < eligibility.size(); ++i) {
                    eligibility[i] += (previous_state[i] / sqrt(2));
                }

                for (int i = 0; i < weight.size(); ++i) {
                    weight[i] += learn_rate*eligibility[i]*td_error;
                }

                bias += learn_rate * td_error;

                break;
            }

            td_error = current_value - previous_value;

            for (int i = 0; i < eligibility.size(); ++i) {
                eligibility[i] += (previous_state[i] / sqrt(2));
            }

            for (int i = 0; i < weight.size(); ++i) {
                weight[i] += learn_rate*eligibility[i]*td_error;
            }

            bias += learn_rate * td_error;

            previous_location = current_location;
            previous_state    = current_state;
            previous_memory   = current_memory;
            previous_signal   = current_signal;
            previous_value    = current_value;

            current_signal = memory[0];          // Turning off the signal

            // Choosing the next move
            Move next_move = nextMove(world[getLeft(current_location)],
                                      current_state,
                                      world[getRight(current_location)],
                                      current_signal,
                                      current_memory,
                                      weight);

            switch(next_move) {
                case Left:
                    current_location = getLeft(current_location);
                    break;
                case Right:
                    current_location = getRight(current_location);
                    break;
                default:
                    int randPick = rand() % 2;
                    if (randPick == 0) {
                        current_location = getLeft(current_location);
                    } else {
                        current_location = getRight(current_location);
                    }
                    break;
            }

            current_location_HRR = world[current_location];
            current_state        = getState(current_location_HRR, current_signal, current_memory);
            current_value        = getValue(current_state, weight) + bias;

            td_error = (current_reward + gamma * current_value) - previous_value;

            for (int i = 0; i < eligibility.size(); ++i) {
                eligibility[i] += (previous_state[i] / sqrt(2));
            }
            
            for (int i = 0; i < weight.size(); ++i) {
                weight[i] += learn_rate*eligibility[i]*td_error;
            }
            
            bias += learn_rate * td_error;

        }
        
    }

    // Print the values
	for (int i = 0; i < NUMBER_SIGNALS; ++i) {
		for (int j = 0; j < NUMBER_SIGNALS; ++j) {
			for (int k = 0; k < WORLD_SIZE; ++k) {
				cout << "Value " << i << " " << j << " " << k << " " << " : " << getValue(getState(world[k], signals[i], memory[j]), weight) << endl;
			}
		}
	}

    return 0;
}

HRR getState(HRR world, HRR signal, HRR memory) {
    return tool.circular_convolution(world, tool.circular_convolution(signal, memory));
}

double getValue(HRR state, HRR& weight) {
    double sum = 0;

    // Dot product of the current state and the weight vector
    for (int i = 0; i < state.size(); ++i) {
        sum += (state[i] * weight[i]);
    }

    return sum;
}

WM memoryUpdate(HRR current_memory, HRR current_signal, HRR nothing_state, HRR current_location_HRR, HRR weight) {
    WM update;
    double current_mem = getValue(getState(current_location_HRR, current_signal, current_memory), weight);
    double _current_sig = getValue(getState(current_location_HRR, current_signal, current_signal), weight);
    double nothin      = getValue(getState(current_location_HRR, current_signal, nothing_state), weight);

    if (current_mem >= _current_sig && current_mem >= nothin) {
        update = current_working_memory;
    } else if (_current_sig >= current_mem && _current_sig >= nothin) {
		update = current_sig;
    } else {
        update = nothing;
    }

    return update;
}

Move nextMove(HRR left, HRR current, HRR right, HRR current_signal, HRR current_memory, HRR weight) {
    Move movement;

    if (getValue(getState(left, current_signal, current_memory), weight) < getValue(getState(right, current_signal, current_memory), weight)) {
        movement = Right;
    } else if (getValue(getState(right, current_signal, current_memory), weight) < getValue(getState(left, current_signal, current_memory), weight)) {
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

int getLeft(int current_location) {
    int left;

    if (current_location == 0) {
        left = WORLD_SIZE - 1;
    } else {
        left = current_location - 1;
    }

    return left;
}

int getRight(int current_location) {
    int right;

    if (current_location == WORLD_SIZE - 1) {
        right = 0;
    } else {
        right = current_location + 1;
    }

    return right;
}

