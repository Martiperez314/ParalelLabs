CC=gcc
CFLAGS=-fopenmp -O2 -Wall -lm
OBJ=argmax

all: $(OBJ)

$(OBJ): argmax.c
	$(CC) $(CFLAGS) argmax.c -o $(OBJ)

clean:
	rm -f $(OBJ) *.o *.out *.err *.txt make.out