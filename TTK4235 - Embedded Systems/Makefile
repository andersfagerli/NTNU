# Which compiler to use
CC = clang-3.8

# Compiler flags go here.
CFLAGS = -g -Wall

# Include path
CURDIR = $(shell pwd)
SRCDIR = $(CURDIR)/src
INC = $(CURDIR)/include

# Linker flags go here.
LDFLAGS = -lcomedi -lm

# list of sources
ELEVSRC = $(wildcard src/*.c) main.c 

# program executable file name.
TARGET = heis

# top-level rule, to compile everything.
all: $(TARGET)

# Define all object files.
ELEVOBJ = $(ELEVSRC:.c=.o)

# rule to link the program
$(TARGET): $(ELEVOBJ)
	$(CC) $^ -o $@ $(LDFLAGS)
	rm -f $(SRCDIR)/*.o $(INC)/*.o *.o *~ 

# Compile: create object files from C source files.
%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@ -I$(INC) 

# rule for cleaning re-compilable files.
clean:
	rm -f $(TARGET) $(ELEVOBJ)
	rm -f $(SRCDIR)/*.o $(INC)/*.o *.o *~

rebuild:	clean all

.PHONY: all rebuild clean


#Martin Korsgaard, 2006
