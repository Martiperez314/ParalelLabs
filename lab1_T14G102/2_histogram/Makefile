CC=gcc
CFLAGS=-O2 -fopenmp -march=native -lm -std=c99
OBJ=histogram


all:
	$(CC) $(OBJ).c random.c -o $(OBJ) $(CFLAGS)

clean:
	rm $(OBJ)

