CC=nvc
CFLAGS=-acc=gpu -gpu=cc90 -Minfo=all -fast 
OBJ=partis_oacc_uni_mem

all:
	$(CC) $(OBJ).c -o $(OBJ) $(CFLAGS)

clean:
	rm $(OBJ)