PRJ=lzwc
CC=gcc
CFLAGS=-O4 -Wall
LDFLAGS=
CPP=cpp
SRCS=$(wildcard *.c)
OBJS=$(patsubst %.c,%.o,$(SRCS))

.PHONY:all dsp clean ed
all:$(PRJ)

$(PRJ):	$(OBJS)
	$(CC) $^ -o $@ $(LDFLAGS)
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
dsp:$(PRJ)
	./$(PRJ)
clean:
	rm -f *.~ $(OBJS) $(PRJ)
ed:
	vim $(PRJ).c

dep:
	sed '/\#\#\# Dependencies:/q' < Makefile > tmp_make
	(for name in *.c;do $(CPP) -MM -MG $$name;done) >> tmp_make
	mv tmp_make Makefile

### Dependencies:
bits_io.o: bits_io.c bits_io.h
lzwc.o: lzwc.c bits_io.h str_table.h
str_table.o: str_table.c str_table.h
util.o: util.c
