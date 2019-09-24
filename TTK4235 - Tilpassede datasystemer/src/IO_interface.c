#include "IO_interface.h"
#include "FSM.h"

#define OPEN 1
#define CLOSE 0

#define ON 1
#define OFF 0

static bool between_floors = false;
static elev_button_type_t call_buttons[N_BUTTONS] = {
    BUTTON_CALL_UP, BUTTON_CALL_DOWN, BUTTON_COMMAND};

floor_t get_floor()
{
  FSM_elevator_t current_state = get_state();
  floor_t current_floor;
  switch (elev_get_floor_sensor_signal())
  {
  case 0:
    current_floor = FIRST_FLOOR;
    break;
  case 1:
    current_floor = SECOND_FLOOR;
    break;
  case 2:
    current_floor = THIRD_FLOOR;
    break;
  case 3:
    current_floor = FOURTH_FLOOR;
    break;
  default:
    current_floor = current_state.floor;
    between_floors = true;
    return current_floor;
  }
  elev_set_floor_indicator(current_floor);
  between_floors = false;
  return current_floor;
}

void start_elevator()
{
  FSM_elevator_t state = get_state();
  elev_motor_direction_t dirn;
  if (state.direction_up)
  {
    dirn = DIRN_UP;
  }
  else
  {
    dirn = DIRN_DOWN;
  }
  elev_set_motor_direction(dirn);
}

void stop_elevator()
{
  FSM_elevator_t current_state = get_state();
  current_state.moving = false;
  set_state(current_state);
  elev_set_motor_direction(DIRN_STOP);
}

void open_doors()
{
  FSM_elevator_t state = get_state();
  if (!state.moving)
  {
    elev_set_door_open_lamp(OPEN);
    state.doors_open = true;
    set_state(state);
  }
}

void close_doors()
{
  FSM_elevator_t current_state = get_state();
  elev_set_door_open_lamp(CLOSE);
  current_state.doors_open = false;
  set_state(current_state);
}

void get_buttons_pushed(bool buttons_pushed_matrix[N_FLOORS][N_BUTTONS])
{
  for (int floor = 0; floor < N_FLOORS; ++floor)
  {
    for (int button = 0; button < N_BUTTONS; ++button)
    {
      buttons_pushed_matrix[floor][button] =
          elev_get_button_signal(call_buttons[button], floor);
    }
  }
}

void clear_stop_signal()
{
  io_clear_bit(STOP);
  elev_set_stop_lamp(OFF);
}

void set_stop_signal()
{
  io_set_bit(STOP);
  elev_set_stop_lamp(ON);
}

bool check_between_floors()
{
  return between_floors;
}