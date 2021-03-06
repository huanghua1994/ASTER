CC = gcc

C_SRCS  = $(wildcard *.c)
C_OBJS  = $(C_SRCS:.c=.c.o)

SLEEF_INSTALL_DIR = $(HOME)/sleef/build-sve/install

DEFS    = -DUSE_SLEEF
INCS    = -I../include -I$(SLEEF_INSTALL_DIR)/include
CFLAGS  = $(INCS) -Wall -g -std=gnu11 -O3 -fopenmp -fPIC $(DEFS)
LDFLAGS = -fopenmp -L$(SLEEF_INSTALL_DIR)/lib64
LIBS    = -lsleef

ifeq ($(shell $(CC) --version 2>&1 | grep -c "gcc"), 1)
AR      = ar rcs
CFLAGS += -march=armv8.2-a+sve -msve-vector-bits=512 -Wno-unused-result -Wno-unused-function
LIBS   += -lm
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
