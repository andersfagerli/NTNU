#include "order.h"

// Initialisation of variables that keep track of orders. Each queue consists of
// bools defining whether a button is pushed in either UP, DOWN or inside elevator
// Using three arrays instead of matrix for readability
static bool queue_down[4] = {0};
static bool queue_up[4] = {0};
static bool queue_inside[N_FLOORS] = {0};

// Matrix with all possible buttons pushed except stop button
static bool buttons_pushed_matrix[N_FLOORS][N_BUTTONS];

// Variables used for timer
static bool timer_started = false;
static time_t start_time = -1;

// Variable used for determining which direction the elevator will prioritise
// floors
static elev_dirn_t prioritised_direction = EMPTY;

// Registers all buttons pushed, updates the correct queues and sets
// prioritised_direction based on buttons pushed
void append_new_orders()
{
  get_buttons_pushed(buttons_pushed_matrix);

  // Loops through all floor buttons
  for (int floor = 0; floor < N_FLOORS; floor++)
  {
    // UP buttons
    if (buttons_pushed_matrix[floor][BUTTON_CALL_UP])
    {
      queue_up[floor] = true;
      elev_set_button_lamp(BUTTON_CALL_UP, floor, 1);
      {
        if (floor != FIRST_FLOOR)
          prioritised_direction = UP;
        else
          prioritised_direction = DOWN;
      }
    }
    // DOWN buttons
    if (buttons_pushed_matrix[floor][BUTTON_CALL_DOWN])
    {
      queue_down[floor] = true;
      elev_set_button_lamp(BUTTON_CALL_DOWN, floor, 1);
      {
        if (floor != FOURTH_FLOOR)
          prioritised_direction = DOWN;
        else
          prioritised_direction = UP;
      }
    }
    // Buttons inside
    if (buttons_pushed_matrix[floor][BUTTON_COMMAND])
    {
      queue_inside[floor] = true;
      elev_set_button_lamp(BUTTON_COMMAND, floor, 1);
      int relative_floor = floor - (int)get_floor();
      if (relative_floor > 0)
      {
        prioritised_direction = UP;
      }
      else if (relative_floor < 0)
      {
        prioritised_direction = DOWN;
      }
    }
  }
}

// Sets the FSM direction and actuates the elevator based on the different queues
// that are set in append_new_orders()
void process_orders()
{
  FSM_elevator_t current_state = get_state();

  if (!(exists_orders_up() || exists_orders_down() || exists_orders_inside()))
  {
    // Sets idle state if no orders
    prioritised_direction = EMPTY;
    stop_elevator();
    return;
  }
  else
  {
    // Orders inside the elevator are prioritised and will determine direction
    if (exists_orders_inside())
    {
      for (int i = 0; i < N_FLOORS; i++)
      {
        if (((get_target_floor() > (int)get_floor() && current_state.direction_up && prioritised_direction == UP && !exists_orders_below()) ||
             (get_target_floor() < (int)get_floor() && !current_state.direction_up && prioritised_direction == DOWN && !exists_orders_above())))
        {
          // If true, keep the same direction as before
          goto end;
        }
        else if (((get_target_floor() > (int)get_floor() && current_state.direction_up && prioritised_direction == DOWN && !exists_orders_below()) ||
                  (get_target_floor() < (int)get_floor() && !current_state.direction_up && prioritised_direction == UP && !exists_orders_above())))
        {
          // Switch the prioritised direction
          switch (prioritised_direction)
          {
          case UP:
            prioritised_direction = DOWN;
            break;
          case DOWN:
            prioritised_direction = UP;
            break;
          default:
            prioritised_direction = EMPTY;
            break;
          }
        }
        else
        {
          current_state.direction_up = !current_state.direction_up;
          switch (prioritised_direction)
          {
          case UP:
            prioritised_direction = DOWN;
            break;
          case DOWN:
            prioritised_direction = UP;
            break;
          default:
            prioritised_direction = EMPTY;
            break;
          }
        }
      }
    }
    else
    // No orders inside
    {
      if (exists_orders_above())
      {
        current_state.direction_up = true;
      }
      else
      {
        current_state.direction_up = false;
      }
    }
    if (!exists_orders_up() && exists_orders_down())
    {
      // Handle 4th floor first if direction is down
      if (queue_down[FOURTH_FLOOR])
        prioritised_direction = UP;
      else
        prioritised_direction = DOWN;
    }
    else if (!exists_orders_down() && exists_orders_up())
    {
      // Handle 1st floor first if direction is up
      if (queue_up[FIRST_FLOOR])
        prioritised_direction = DOWN;
      else
        prioritised_direction = UP;
    }
  }

end:
  // Reverse direction if in end floors
  if (get_floor() == FIRST_FLOOR)
  {
    current_state.direction_up = true;
  }
  else if (get_floor() == FOURTH_FLOOR)
  {
    current_state.direction_up = false;
  }
  set_state(current_state);

  // Start elevator only if doors are closed
  if (!current_state.doors_open)
  {
    start_elevator();
  }
}

// Stops the elevator if the conditions for a stop at the current floor is true
// and activates door and timer
void stop_at_floor()
{
  FSM_elevator_t current_state = get_state();

  // Variable for determining the last floor to stop at in the prioritised
  // direction
  floor_t target_floor = get_target_floor();
  if ((target_floor == get_floor() ||
       ((current_state.direction_up || !exists_orders_down() || prioritised_direction == UP) && queue_up[get_floor()] && (get_target_floor_up() == (int)get_floor())) ||
       ((!current_state.direction_up || !exists_orders_up() || prioritised_direction == DOWN) && queue_down[get_floor()] && (get_target_floor_down() == (int)get_floor())) ||
       queue_inside[get_floor()] ||
       timer_started) &&
      !check_between_floors())
  {
    stop_elevator();
    current_state.floor = get_floor();
    clear_button_lights(current_state.floor);
    set_state(current_state);

    // Resets all orders on the current floor
    queue_up[get_floor()] = false;
    queue_down[get_floor()] = false;
    queue_inside[get_floor()] = false;

    // Activates doors and starts timer
    open_and_close_doors();
  }
}

// Returns true if there exists orders with direction up or inside elevator
bool exists_orders_up()
{
  for (int i = 0; i < N_FLOORS; i++)
  {
    if (queue_up[i] || queue_inside[i])
    {
      return true;
    }
  }
  return false;
}

// Returns true if there exists orders with direction down or inside elevator
bool exists_orders_down()
{
  for (int i = 0; i < N_FLOORS; i++)
  {
    if (queue_down[i] || queue_inside[i])
    {
      return true;
    }
  }
  return false;
}

// Return true if there exists orders inside elevator
bool exists_orders_inside()
{
  for (int i = 0; i < N_FLOORS; i++)
  {
    if (queue_inside[i])
    {
      return true;
    }
  }
  return false;
}

// Returns true if there exists orders above the current floor
bool exists_orders_above()
{
  int current_floor = (int)get_floor();
  for (int i = 3; i > current_floor; i--)
  {
    if (queue_up[i] || queue_down[i])
    {
      return true;
    }
  }
  return false;
}

// Returns true if there exists orders below the current floor
bool exists_orders_below()
{
  int current_floor = (int)get_floor();
  for (int i = 0; i < current_floor; i++)
  {
    if ((queue_up[i]) || (queue_down[i]))
    {
      return true;
    }
  }
  return false;
}

// Returns the last floor to stop at for orders inside
floor_t get_target_floor()
{
  FSM_elevator_t current_state = get_state();
  if (current_state.direction_up)
  {
    for (int i = 3; i >= 0; i--)
    {
      if (queue_inside[i])
      {
        return (floor_t)i;
      }
    }
  }
  else
  {
    for (int i = 0; i < 4; i++)
    {
      if (queue_inside[i])
      {
        return (floor_t)i;
      }
    }
  }

  // If no orders an invalid floor is returned
  return -2;
}

// Returns the last floor to stop at in upward direction
floor_t get_target_floor_up()
{
  FSM_elevator_t current_state = get_state();
  if (current_state.direction_up)
  {
    for (int i = 0; i < 4; i++)
    {
      if (queue_up[i])
      {
        return (floor_t)i;
      }
    }
  }
  else
  {
    for (int i = 0; i < 4; i++)
    {
      if (queue_up[i])
      {
        return (floor_t)i;
      }
    }
  }

  return -2;
}

// Returns the last floor to stop at in downward direction
floor_t get_target_floor_down()
{
  FSM_elevator_t current_state = get_state();
  if (current_state.direction_up)
  {
    for (int i = 3; i >= 0; i--)
    {
      if (queue_down[i])
      {
        return (floor_t)i;
      }
    }
  }
  else
  {
    for (int i = 0; i < 4; i++)
    {
      if (queue_down[i])
      {
        return (floor_t)i;
      }
    }
  }

  return -2;
}

// Opens doors and closes after 3 seconds
void open_and_close_doors()
{
  if (!timer_started)
  {
    open_doors();
    start_timer();
    timer_started = true;
  }
  if (timeout())
  {
    timer_started = false;
    close_doors();
  }
}

// Stops elevator, resets orders and open doors if elevators is on a floor
void stop_button_pushed()
{
  set_stop_signal();
  stop_elevator();
  reset_orders();
  if (!check_between_floors())
  {
    while (elev_get_stop_signal())
    {
      open_doors();
    }
    timer_started = false;
    open_and_close_doors();
  }
  clear_stop_signal();
}

// Resets all orders
void reset_orders()
{
  for (int floor = 0; floor < N_FLOORS; ++floor)
  {
    clear_button_lights(floor);
    queue_up[floor] = 0;
    queue_down[floor] = 0;
    queue_inside[floor] = 0;
  }
}

// Starts timer
void start_timer()
{
    start_time = time(0);
}

// Bool function for determining if 3 seconds have passed
bool timeout()
{
  if (start_time < 0)
  {
    return false;
  }

  time_t now = time(0);
  if ((now - start_time) > 3)
  {
    start_time = -1;
    return true;
  }
  else
  {
    return false;
  }
}

// Clears the button lights at a given floor
void clear_button_lights(int floor)
{
  if (queue_down[floor])
    elev_set_button_lamp(BUTTON_CALL_DOWN, floor, 0);
  if (queue_up[floor])
    elev_set_button_lamp(BUTTON_CALL_UP, floor, 0);
  if (queue_inside[floor])
    elev_set_button_lamp(BUTTON_COMMAND, floor, 0);
}
