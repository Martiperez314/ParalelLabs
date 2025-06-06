CC=nvc
CFLAGS=-acc=gpu -gpu=cc90 -Minfo=all -fast 
OBJ=partis_oacc_managed 

all:
	$(CC) $(OBJ).c -o $(OBJ) $(CFLAGS)

clean:
	rm $(OBJ)