#include "FSM.h"

// Static state variable that keeps track of the current state
static FSM_elevator_t state;

// Gets a copy of the current state
FSM_elevator_t get_state() { return state; }

// Sets the state
void set_state(FSM_elevator_t new_state) { state = new_state; }

// Initialisation function to set the elevator into a defined state
void FSM_init()
{
  state.direction_up = true;
  state.moving = false;
  state.floor = FIRST_FLOOR;
  state.doors_open = false;
  state.stop_button_pushed = false;
  clear_stop_signal();
  while (get_floor() == state.floor)
  {
    start_elevator();
  }
  state.floor = get_floor();
  stop_elevator();
}
