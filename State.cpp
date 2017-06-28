/* ========================================================================
   $File: State.cpp
   $Date: $
   $Revision: $
   $Creator: Bishoy Boktor $
   $Notice: (C) Copyright 2017. All Rights Reserved. $
   ======================================================================== */

#include <State.h>

State::State() {
    reward = 0;
    value = 0;
    eligibility = 0;
    index = 0;
}

State::State(double r, double v, int i) {
    reward = r;
    value = v;
    eligibility = 0;
    index = i;
}

double State::getReward() {
    return reward;
}

double State::getValue() {
    return value;
}

double State::getEligibility() {
  return eligibility;
}

int State::getIndex() {
    return index;
}

void State::setReward(double r) {
    reward = r;
}

void State::setValue(double v) {
    value = v;
}

void State::setEligibility() {
  eligibility = 1;
}

void State::setIndex(int i) {
    index = i;
}

void State::resetEligibility() {
  eligibility = 0;
}

void State::updateEligibility(double lambda) {
  eligibility *= lambda;
}
