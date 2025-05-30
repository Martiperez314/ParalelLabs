CC = gcc
CFLAGS = -O2 -fopenacc -lm
TARGET = partis_oacc_uni_mem
SRC = partis_oacc_uni_mem.c


all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(SRC) -o $(TARGET) $(CFLAGS)

clean:
	rm -f $(TARGET) *.o out/* *.nsys-rep