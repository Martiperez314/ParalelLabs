CC = nvc
CFLAGS = -O3 -Wall -acc -gpu=managed
TARGET = partis_oacc_uni_mem
SRC = partis_oacc_uni_mem.c

all:
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) -lm

clean:
	-rm -f $(TARGET) *.nsys-rep partis_uni_mem.out partis_uni_mem.err
	-rm -rf out