TARGET=main
OBJECTS=util.o matmul.o

CPPFLAGS=-O3 -Wall -march=native -mavx2 -mfma -fopenmp -mavx512f
LDLIBS=-lm -lpthread -lmpi -lmpi_cxx

CC=mpicc

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
