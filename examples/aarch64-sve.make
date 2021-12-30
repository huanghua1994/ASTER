CC = gcc
#CC = armclang

# If ARM Performance Libraries are available, use it for vectorized math functions
USE_ARM_PL = 1

C_SRCS  = $(wildcard *.c)
C_OBJS  = $(C_SRCS:.c=.c.o)

DEFS    = 
INCS    = -I../include
CFLAGS  = $(INCS) -Wall -g -std=gnu11 -O3 -fopenmp -fPIC $(DEFS)
LDFLAGS = -fopenmp
LIBS    = 

ifeq ($(shell $(CC) --version 2>&1 | grep -c "gcc"), 1)
AR      = ar rcs
CFLAGS += -march=armv8.2-a+sve -msve-vector-bits=512 -Wno-unused-result -Wno-unused-function
LIBS   += -lm
endif

ifeq ($(shell $(CC) --version 2>&1 | grep -c "Arm C"), 1)
AR      = ar rcs
CFLAGS += -march=armv8.2-a+sve -msve-vector-bits=512 -Wno-unused-result -Wno-unused-function
LIBS   += -lm
endif

ifeq ($(strip $(USE_ARM_PL)), 1)
LIBS   += -larmpl
else
DEFS   += -DNEON_LOOP_VEC_MATH
endif

# Delete the default old-fashion double-suffix rules
.SUFFIXES:

.SECONDARY: $(C_OBJS)

EXES = $(C_SRCS:.c=.exe)

default: $(EXES)

%.exe: %.c.o $(LIB)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

%.c.o: %.c
	$(CC) $(CFLAGS) -c $^ -o $@

clean:
	rm $(C_OBJS) $(EXES)
