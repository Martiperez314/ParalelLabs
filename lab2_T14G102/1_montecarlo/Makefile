CC=gcc
CFLAGS=-O2 -fopenmp -march=native -lm -std=c99
OBJ=montecarlo


all:
	$(CC) main.c $(OBJ).c -o $(OBJ) $(CFLAGS)

clean:
	rm $(OBJ)
