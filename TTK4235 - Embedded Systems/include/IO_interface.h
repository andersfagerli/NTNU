#ifndef IO_INTERFACE_H
#define IO_INTERFACE_H

// Include macros of not already defined
#ifndef N_BUTTONS
#define N_BUTTONS 3
#endif

// Include necessary libraries
#include "elev.h"
#include "channels.h"
#include "io.h"

#include <stdbool.h>
#include <stdio.h>

// Enumeration for each floor to increase readability
typedef enum {
  BETWEEN_FLOORS = -1,
  FIRST_FLOOR = 0,
  SECOND_FLOOR = 1,
  THIRD_FLOOR = 2,
  FOURTH_FLOOR = 3
} floor_t;

// Function that returns what floor the elevator is on.
// If the elevator is between floors the function returns the previous floor
// the elevator was on and sets "between floors" to true
floor_t get_floor();

// Function that starts the elevator in a direction given by the state.
void start_elevator();

// Function that stops the elevator
void stop_elevator();

// Function that sets the state and turns on the "open doors" lamp
void open_doors();

// Sets the state and turns off the lamp
void close_doors();

// Gets which buttons are pushed
void get_buttons_pushed(bool buttons_pushed_matrix[N_FLOORS][N_BUTTONS]);

// Clears stop signal
void clear_stop_signal();

// Sets stop signal
void set_stop_signal();

// Return whether the elevator is between floors
bool check_between_floors();

#endif
