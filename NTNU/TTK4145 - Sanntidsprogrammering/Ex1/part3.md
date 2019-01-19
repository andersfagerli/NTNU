# Reasons for concurrency and parallelism


To complete this exercise you will have to use git. Create one or several commits that adds answers to the following questions and push it to your groups repository to complete the task.

When answering the questions, remember to use all the resources at your disposal. Asking the internet isn't a form of "cheating", it's a way of learning.

 ### What is concurrency? What is parallelism? What's the difference?
 > Parallelism is when several instances of a program are run at the same time, but run on different data. There is no requirement of coordination between the instances, a contrast to concurrency. Concurrency is when several instances of a program run within the same time frame, but in the course of execution they must coordinate with each other in order for the program as whole to execute correctly. Parellelism is thus a way to implement concurrency. An example may be a search engine, which may use several threads for searching different URLs at the same time. Several instances of the search algorithm are thus used, but on different data (parallelism). If the algorithm also distributes URLs to the different instances so that no instance searches for already searched URLs, the algorithm is concurrent, as it demands some form of coordination between program instances.
 
 ### Why have machines become increasingly multicore in the past decade?
 > To allow for faster and more efficient execution of tasks, as the size of datasets in computing problems have increased the past decade. We have reached the point where a single processor cannot run any faster, and multicore processors solves this problem. 
 
 ### What kinds of problems motivates the need for concurrent execution?
 (Or phrased differently: What problems do concurrency help in solving?)
 > Real-time problems. Ex. an operating system, which needs to handle several tasks at the same time. 
 
 ### Does creating concurrent programs make the programmer's life easier? Harder? Maybe both?
 (Come back to this after you have worked on part 4 of this exercise)
 > Both easier and harder, depending on the problem. There is more complexity in terms of syntax and possible faults, but it may ease the design of the program.
 
 ### What are the differences between processes, threads, green threads, and coroutines?
 > A process is needed to execute a program, and may consist of several threads in order to do so. Threads share memory space, while processes run in separate memory spaces. Green threads are scheduled by a virtual machine, as opposed to standard threads which are scheduled by the native operating system. A coroutine is a control structure where control is passed between two different routines, so that one routine may run for a given amount of time and stop to pass control to another routine, and then return where it left off when control is given back.
 
 ### Which one of these do `pthread_create()` (C/POSIX), `threading.Thread()` (Python), `go` (Go) create?
 > They create a new thread. Go creates and handles its own threads, and may therefore seem as a green thread.
 
 ### How does pythons Global Interpreter Lock (GIL) influence the way a python Thread behaves?
 > The GIL ensures that multiple threads can't access multiple cores of the processer, so threads can't run in parallel. The program will as result only use one core at any given time.
 
 ### With this in mind: What is the workaround for the GIL (Hint: it's another module)?
 > GIL is only implemented on CPython, using a different implementation of Python is a workaround. Examples are Jython, IronPython. Another workaround is importing the library "multiprocessing".
 
 ### What does `func GOMAXPROCS(n int) int` change? 
 > The function changes the maximum number of cores that can execute simultaneously in a Go program.
