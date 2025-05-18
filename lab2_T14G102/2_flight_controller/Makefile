CC=mpicc
CFLAGS=-O3 -march=native -lm -std=c99
OBJ = fc_mpi

all:
	$(CC) fc_mpi.c auxiliar.c -o fc_mpi $(CFLAGS)

clean:
	rm $(OBJ)
S