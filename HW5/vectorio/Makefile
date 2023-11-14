TARGET=main
OBJECTS=timer.o util.o vec_add.o

CC=gcc
CFLAGS=-std=c11 -O3 -Wall -fopenmp
LDLIBS=-lm

# OpenCL settings
OBJECTS+=opencl_util.o
CFLAGS+=-I/usr/local/cuda/include
LDLIBS+=-L/usr/local/cuda/lib64 -lOpenCL

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)

