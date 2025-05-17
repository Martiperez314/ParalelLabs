CC = mpicc
CFLAGS = -O3 -march=native -std=c99 -Wall
LIBS = -lm

all: fc_seq fc_mpi

fc_seq: fc_seq.c auxiliar.c
	$(CC) $(CFLAGS) -o fc_seq fc_seq.c auxiliar.c $(LIBS)

fc_mpi: fc_mpi.c auxiliar.c
	$(CC) $(CFLAGS) -o fc_mpi fc_mpi.c auxiliar.c $(LIBS)

clean:
	rm -f fc_seq fc_mpi
