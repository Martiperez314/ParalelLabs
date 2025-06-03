CC = nvc
CFLAGS = -O3 -Wall -acc -gpu=cc90
TARGET = partis_oacc_async
SRC = partis_oacc_async.c

all:
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) -lm

clean:
	-rm -f $(TARGET) *.nsys-rep partis_async.out partis_async.err
	-rm -rf out