CC = gcc
CFLAGS = -O3 -Wall
TARGET = partis_seq
SRC = partis_seq.c

all:
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) -lm

clean:
	rm -f $(TARGET) *.o partis_seq.out partis_seq.err
	rm -rf out