CC=mpicc
CFLAGS=-O2 -fopenmp -march=native -lm -std=c99
OBJ=montecarlo

all:
	$(CC) $(OBJ).c -o $(OBJ) $(CFLAGS)

clean:
	rm $(OBJ)
