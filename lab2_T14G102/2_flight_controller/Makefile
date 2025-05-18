CC      = mpicc
CFLAGS  = -O3 -lm -std=c99
LIBS    =

.PHONY: all seq mpi clean

all: seq mpi

# alias so "make mpi" actually builds fc_mpi
mpi: fc_mpi

# build sequential (if you ever need it)
seq: fc_seq

fc_seq: fc_seq.c auxiliar.c
        $(CC) $(CFLAGS) -o $@ fc_seq.c auxiliar.c $(LIBS)

# build the MPI executable
fc_mpi: fc_mpi.c auxiliar.c
        $(CC) $(CFLAGS) -o $@ fc_mpi.c auxiliar.c $(LIBS)

clean:
        rm -f fc_seq fc_mpi

