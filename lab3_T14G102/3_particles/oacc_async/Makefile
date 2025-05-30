CC = nvc
CFLAGS = -O3 -Wall
TARGET = partis_oacc_async
SRC = partis_oacc_async.c

# Kompilieren mit Unified Memory (automatisches Speicherhandling durch Compiler/GPU)
managed:
	$(CC) $(CFLAGS) -ta=tesla:managed -o $(TARGET) $(SRC) -lm

# Kompilieren mit manuellem Speicherhandling
manual:
	$(CC) $(CFLAGS) -ta=tesla:cc90 -o $(TARGET) $(SRC) -lm

# Aufräumen: Entfernt die Binärdatei und Ausgabeverzeichnis
clean:
	-rm -f $(TARGET) *.nsys-rep
	-rm -rf out