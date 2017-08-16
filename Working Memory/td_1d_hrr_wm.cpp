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

typedef vector<double> HRR;

enum Move {
    Left,
    Right
};

enum WM {
    current_working_memory,
    current_signal,
    nothing
};
    

HRR getState(const DFTTool& tool, HRR world, HRR signal, HRR memory);
double getValue(HRR state, HRR& weight);
Move nextMove(const DFTTool& tool, HRR left, HRR current, HRR right, HRR current_signal, HRR current_memory, HRR weight);
Move randMove();
int getLeft(int current_location);
int getRight(int current_location);
WM memoryUpdate(const DFTTool& tool, HRR current_memory, HRR current_signal, HRR nothing_state, HRR current_location_HRR, HRR weight);

int main() {
    int world_size          = 20;
    int number_signals      = 3;
    int number_episodes     = 1000;
    int steps               = 100;
    
    double default_reward   = 0;
    double goal_reward      = 1;
    int rand_goal1          = rand() % (world_size);
    int rand_goal2          = rand() % (world_size);
    double previous_reward;
    
    int hrr_size            = 1024;
    int bias                = 0;
    double gamma            = 0.9;
    double learn_rate       = 0.3;
    double lambda           = 0.5;
    double td_error;

    vector<HRR> world(world_size);
    vector<HRR> signals(number_signals);   // The signals present in environment
    vector<HRR> memory(number_signals);    // The signals remembered by the agent
    HRR weight(hrr_size);
    vector<HRR> eligibility(hrr_size);
    int reward[number_signals][world_size];
    
    DFTTool tool(hrr_size);
    
    for (int i = 0; i < world_size; ++i) {
        world[i].pushback(tool.hrr());
    }

    signals[0].pushback(tool.ihrr());
    memory[0].pushback(tool.ihrr());
    for (int i = 1; i <= number_signals; ++i) {
        signals[i].pushback(tool.hrr());
        memory[i].pushback(tool.hrr());
    }

    for (int i = 0; i < number_signals; ++i) {
        for (int j = 0; j < world_size; ++j) {
            reward[i][j] = default_reward;
        }
    }

    // Choosing random goals
    reward[1][rand_goal1] = goal_reward;
    reward[2][rand_goal2] = goal_reward;

    for (int episode = 1; episode <= number_episode; ++episode) {
        /* for (int i = 0; i < hrr_size; ++i) {
            eligibiltiy[i].pushback(0);
            } */
        std::fill(eligibility.begin(), eligibility.end(), 0);

        int current_task               = (rand() > RAND_MAX/2) ? 1 : 2; 
        int current_location           = rand() % (world_size);
        HRR current_location_HRR       = world[current_location];
        HRR current_memory             = memory[0];
        HRR current_signal             = signals[current_task];

        for (int timestep = 0; timestep < steps; ++timestep) {
            int current_reward         = reward[current_tast][current_location];
            HRR current_state          = getState(tool, current_location_HRR, current_signal, current_memory);
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
            WM memory_update = memoryUpdate(tool, current_memory, current_signal, nothing_state, current_location_HRR, weight);

            // Making the actual memory update
            switch(memory_update) {
                case current_working_memory:
                    break;                // In the current_memory case we don't need to do anything 
                case current_signal:
                    current_memory = current_signal;
                    break;
                case nothing:
                    current_memory = nothing_state;
                    break;
                default:
                    // make default current_working_memory
                    break;
            }

            current_state = getState(tool, current_location_HRR, current_signal, current_memory);
            current_value = getValue(current_state, weight) + bias;


            if (reward[current_task][current_location] == goal_reward) {
                td_error = current_value - previous_value;

                for (int i = 0; i < eligibility.size(); ++i) {
                    eligibility[i] += (previous_state / sqrt(2));
                }

                for (int i = 0; i < weight.size(); ++i) {
                    weight[i] += lrate*eligibility[i]*td_error;
                }

                bias += lrate * td_error;
                
                break;
            }

            td_error = current_value - previous_value;

            for (int i = 0; i < eligibility.size(); ++i) {
                eligibility[i] += (previous_state / sqrt(2));
            }
            
            for (int i = 0; i < weight.size(); ++i) {
                weight[i] += lrate*eligibility[i]*td_error;
            }
            
            bias += lrate * td_error;

            previous_location = current_location;
            previous_state    = current_state;
            previous_memory   = current_memory;
            previous_signal   = current_signal;
            previous_value    = current_value;
            
            current_signal = memory[0];          // Turning off the signal

            // Choosing the next move
            Move next_move = nextMove(tool,
                                      world[current_task][getLeft(current_location)],
                                      current_state,
                                      world[current_task][getRight(current_location)],
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

            current_location_HRR = world[current_task][current_location];
            current_state        = getState(tool, current_location_HRR, current_signal, current_memory);
            current_value        = getValue(current_state, weight) + bias;

            td_error = (current_reward + gamma * current_value) - previous_value;

            for (int i = 0; i < eligibility.size(); ++i) {
                eligibility[i] += (previous_state / sqrt(2));
            }
            
            for (int i = 0; i < weight.size(); ++i) {
                weight[i] += lrate*eligibility[i]*td_error;
            }
            
            bias += lrate * td_error;

        }
        
    }

    // Print the values

    return 0;
}

HRR getState(const DFTTool& tool, HRR world, HRR signal, HRR memory) {
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

WM memoryUpdate(const DFTTool& tool, HRR current_memory, HRR current_signal, HRR nothing_state, HRR current_location_HRR, HRR weight) {
    WM update;
    double current_mem = getValue(getState(tool, current_location_HRR, current_signal, current_memory), weight);
    dobule current_sig = getValue(getState(tool, current_location_HRR, current_signal, current_signal), weight);
    double nothin      = getValue(getState(tool, current_locatoin_HRR, current_signal, nothing_state), weight);

    if (current_mem >= current_sig && current_mem >= nothin) {
        update = current_working_memory;
    } else if (current_sig >= current_mem && current_sig >= nothin) {
        update = current_signal;
    } else {
        update = nothing;
    }

    return update;
}

Move nextMove(const DFTTool& tool, HRR left, HRR current, HRR right, HRR current_signal, HRR current_memory, HRR weight) {
    Move movement;

    if (getValue(getState(tool, left, current_signal, current_memory), weight) < getValue(getState(tool, right, current_signal, current_memory), weight)) {
        movement = Right;
    } else if (getValue(getState(tool, right, current_signal, current_memory), weight) < getValue(getState(tool, left, current_signal, current_memory), weight)) {
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
        left = world_size - 1;
    } else {
        left = current_locatoin - 1;
    }

    return left;
}

int getRight(int current_location); {
    int right;

    if (current_location == world_size - 1) {
        right = 0;
    } else {
        right = current_locatoin + 1;
    }

    return right;
}

