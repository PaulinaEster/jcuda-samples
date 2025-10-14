SHELL=/bin/sh

# WORKLOAD
WORKLOAD=A

# DEBUG FLAG
DEBUG=OFF
DEBUG_FLAG=NO_DEBUG
ifeq ($(DEBUG),ON)
	DEBUG_FLAG=DEBUG
endif

# TIMER FLAG
TIMER=OFF
TIMER_FLAG=NO_TIMER
ifeq ($(TIMER),ON)
	TIMER_FLAG=TIMER
endif

include ../config/make.def

all: matrix_multiplication

matrix_multiplication:
	$(CCOMPILER) matrix_multiplication.c $(CFLAGS) -DWORKLOAD_$(WORKLOAD) -D$(DEBUG_FLAG) -D$(TIMER_FLAG) -o matrix_multiplication.$(WORKLOAD).exe

clean:
	- rm -f *.o *~ matrix_multiplication.*.exe
