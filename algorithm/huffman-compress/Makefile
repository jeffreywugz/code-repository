PRJ=hc
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
hc.o: hc.c util.h bits_io.h hfm_tree.h
heap.o: heap.c util.h heap.h hfm_tree.h
hfm_tree.o: hfm_tree.c util.h prio_queue.h heap.h hfm_tree.h
prio_queue.o: prio_queue.c prio_queue.h heap.h hfm_tree.h
util.o: util.c
