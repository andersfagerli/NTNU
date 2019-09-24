#ifndef ELEV_2_H
#define ELEV_2_h

#include "elev2.h"

int elev_get_floor_sensor_signal() {
  int floor;
  printf("Type in floor: ");
  scanf("%d", &floor);
  return (floor);
};

void elev_set_motor_direction(elev_motor_direction_t dirn) {
  switch (dirn) {
  case DIRN_DOWN:
    printf("Direction set to \"down\"\n");
    break;
  case DIRN_STOP:
    printf("Direction set to \"stop\"\n");
    break;
  case DIRN_UP:
    printf("Direction set to \"down\"\n");
    break;
  default:
    break;
  }
}

void elev_set_door_open_lamp(int value) {}

int elev_get_button_signal(elev_button_type_t button, int floor) {}

#endif