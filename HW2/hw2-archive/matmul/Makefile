TARGET=main
OBJECTS=util.o matmul.o

CPPFLAGS=-O3 -Wall -march=native -mavx512f -mavx2 -mfma -fopenmp 
LDFLAGS=-lm

CC=gcc

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
