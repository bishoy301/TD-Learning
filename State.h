/* ========================================================================
   $File: State.h
   $Date: $
   $Revision: $
   $Creator: Bishoy Boktor $
   $Notice: (C) Copyright 2017. All Rights Reserved. $
   ======================================================================== */

#if !defined(STATE_H)
#define STATE_H

class State {
private:
    double reward;
    double value;
    int index;
    double eligibility;

public:
    State();
    State(double r, double v, int i);
    
    //Accessors
    double getReward();
    double getEligibility();
    double getValue();
    int getIndex();

    //Mutators
    void setReward(double r);
    void setEligibility();
    void setValue(double v);
    void setIndex(int i);

    //Eligibility Trace methods
    void resetEligibility();
    void updateEligibility(double lambda);
};
    
#endif
