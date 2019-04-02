# Mutex and Channel basics

### What is an atomic operation?
> An operation where read and write happens during the same data transmission, meaning no other operation can read or write to the specific memory space while the atomic operation is operating on it.
### What is a semaphore?
> A variable that controls access to a common resource in concurrent systems, and is typically an integer greater than or equal to zero. Uses the methods wait(S) and signal(S) to pass control. wait(S) will decrement the value of S if it is greater than zero, and delay the task otherwise. signal(S) will increment the value of S.

### What is a mutex?
> A mutex is an object used to pass control between different threads working on a shared resource, such that they don't operate on it simultaneously. A mutex may be locked so only one thread may operate on the resource, and later be unlocked for other threads to use it.

### What is the difference between a mutex and a binary semaphore?
> A mutex gives ownership to the common resource.

### What is a critical section?
> A critical section of a program is the section in which shared resources between several programs or threads are being operated on.

### What is the difference between race conditions and data races?
 > Race conditions are when a shared resource is changed by two or several threads, leading to a fault in the program behavior. A data race is when several threads access a shared resource, but it does not result in a fault in program behavior. A race condition may happen when one or both of two threads are writing, and the other is reading. Under a data race, the threads will not read.

### List some advantages of using message passing over lock-based synchronization primitives.
> Message passing scales better

### List some advantages of using lock-based synchronization primitives over message passing.
> Efficient
