#ifndef order_H
#define order_H

#include <stdlib.h>
#include <time.h>

#include "FSM.h"
#include "IO_interface.h"

// Enumerator for direction of elevator
typedef enum {
    UP = 0,
    DOWN = 1,
    EMPTY = 2
} elev_dirn_t;

// Registers all buttons pushed, updates the correct queues and sets
// prioritised_direction based on buttons pushed
void append_new_orders();

// Sets the FSM direction and actuates the elevator based on the different queues
// that are set in append_new_orders()
void process_orders();

// Stops the elevator if the conditions for a stop at the current floor is true
// and activates door and timer
void stop_at_floor();

// Returns true if there exists orders with direction up or inside elevator
bool exists_orders_up();

// Returns true if there exists orders with direction down or inside elevator
bool exists_orders_down();

// Return true if there exists orders inside elevator
bool exists_orders_inside();

// Returns true if there exists orders above the current floor
bool exists_orders_above();

// Returns true if there exists orders below the current floor
bool exists_orders_below();

// Returns the last floor to stop at for orders inside
floor_t get_target_floor();

// Returns the last floor to stop at in upward direction
floor_t get_target_floor_up();

// Returns the last floor to stop at in downward direction
floor_t get_target_floor_down();

// Opens doors and closes after 3 seconds
void open_and_close_doors();

// Stops elevator, resets orders and open doors if elevators is on a floor
void stop_button_pushed();

// Resets all orders
void reset_orders();

// Starts timer
void start_timer();

// Bool function for determining if 3 seconds have passed
bool timeout();

// Clears the button lights at a given floor
void clear_button_lights(int floor);

#endif
