CC=gcc
CFLAGS=-fopenmp
LDFLAGS=-lm

SOURCE := $(wildcard *.c)
TARGETS = $(SOURCE:.c=)

all: $(TARGETS)

$(TARGETS): $(SOURCE)
	$(CC) $(CFLAGS) $@.c -o $@.out $(LDFLAGS)

clean:
	rm $(addsuffix ".out", $(TARGETS))
