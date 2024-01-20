CC=gcc
CFLAGS=
SOURCES=$(filter-out square.c, $(wildcard *.c))
OBJECTS=$(SOURCES:.c=.out)

all: $(OBJECTS)

%.out: %.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(OBJECTS)
