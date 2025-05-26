CC=nvcc
CFLAGS=-O2 -lm -Xcompiler -Wall
OBJ=vecadd_cuda

all:
	$(CC) $(CFLAGS) $(OBJ).cu -o $(OBJ)

clean:
	rm $(OBJ)