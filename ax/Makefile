PRJ=ax
CC=gcc
CPP=cpp
CFLAG=-Wall -Werror -g
LDFLAG=
.PHONY: all run clean  dep

SRCS= $(wildcard *.c)
OBJS= $(patsubst %.c,%.o,$(SRCS))

all: $(PRJ)
$(PRJ): $(OBJS)
	$(CC) -o $@ $(LDFLAG) $^
%.o: %.c
	$(CC) -c  $(CFLAG) $<
run: $(PRJ)
	./$(PRJ)
clean:
	rm -rf $(PRJ) *.o *~ .dep
dep:
	(for f in $(SRCS);do echo -n "";$(CPP) -M -MM -MG $$f;done) >> .dep

### Dependencies:
-include .dep
