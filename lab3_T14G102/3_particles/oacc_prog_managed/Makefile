CC = nvc
CFLAGS = -O3 -Wall -acc -gpu=managed
SRC= partis_oacc_managed.c
OUT= partis_sim

all:
	$(CC) $(CFLAGS) $(SRC) -o $(OUT)

clean:
	rm -f $(OUT) *.o *.out *.err *.rep out/*.csv
