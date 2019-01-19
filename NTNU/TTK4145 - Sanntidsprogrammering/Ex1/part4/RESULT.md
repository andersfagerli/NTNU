# Results from foo.c, foo.go and foo.py

> In each program, we create two threads that act on a common variable i by respectively increasing and decreasing its value by 1 000 000. Since we in total increase as much as we decrease, the expected value might be 0. The value after running is however some large random number that is either negative or positive.
This is a result of the threads not being synchronized, meaning they have tried changing the common variable i at the same time.
