#include "order.h"

int main() {
  printf("Hallo\n");
  FSM_init();
  FSM_elevator_t current_state = get_state();
  printf("%i\n", current_state.floor);
  return 0;
}