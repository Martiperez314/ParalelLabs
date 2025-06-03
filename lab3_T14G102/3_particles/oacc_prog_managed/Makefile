CC = nvc
CFLAGS = -O3 -Wall -acc -gpu=cc90
TARGET = partis_oacc_managed
SRC = partis_oacc_managed.c

all:
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) -lm

clean:
	-rm -f $(TARGET) *.nsys-rep partis_managed.out partis_managed.err
	-rm -rf out