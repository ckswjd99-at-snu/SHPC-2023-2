TARGET=main
OBJECTS=util.o matmul.o

CPPFLAGS=-O3 -Wall -march=native -mavx2 -mfma -fopenmp -mno-avx512f -I/usr/local/cuda/include
LDFLAGS=-L/usr/local/cuda/lib64
LDLIBS=-lm -pthread -lOpenCL

CC=gcc

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
