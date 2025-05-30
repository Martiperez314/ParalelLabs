CC = gcc
CFLAGS = -O3 -Wall
TARGET = partis_seq

all: $(TARGET)

$(TARGET): partis_seq.c
	$(CC) $(CFLAGS) partis_seq.c -o $(TARGET) -lm

clean:
	rm -f $(TARGET) *.o
	rm -rf out video