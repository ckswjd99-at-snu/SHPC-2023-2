TARGET=main
OBJECTS=util.o monte_carlo.o

CPPFLAGS=-O3 -Wall -march=native -mavx2 -mfma -mavx512f -fopenmp
LDLIBS=-lm -lmpi -lmpi_cxx -lpthread

CC=mpicc

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
