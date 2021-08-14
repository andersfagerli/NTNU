#ifndef FSM_H
#define FSM_H

// Include necessary libraries
#include "IO_interface.h"

// Struct used to define the state of the elevator
typedef struct
{
  bool direction_up;
  bool moving;
  floor_t floor;
  bool doors_open;
  bool stop_button_pushed;
} FSM_elevator_t;

// Gets a copy of the current state
FSM_elevator_t get_state();

// Sets the state
void set_state(FSM_elevator_t state);

// Initialisation function to set the elevator into a defined state
void FSM_init();

#endif
