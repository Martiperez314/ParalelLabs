CC=nvc
CFLAGS=-acc=gpu -gpu=cc90 -Minfo=all -lm
OBJ=partis_oacc_async

all:
	$(CC) $(CFLAGS) $(OBJ).c -mp -o $(OBJ)

clean:
	rm $(OBJ)