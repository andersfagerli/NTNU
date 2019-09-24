#include "order.h"

int main()
{
  //     Initialize hardware
  if (!elev_init())
  {
    printf("Unable to initialize elevator hardware!\n");
    return 1;
  }
  // Initialize the state of the elevator
  FSM_init();

  while (1)
  {
    if (elev_get_stop_signal())
    {
      stop_button_pushed();
    }
    append_new_orders();
    process_orders();
    stop_at_floor();
  }
  return 0;
}
