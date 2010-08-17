PRJ=drawing
CC=gcc
CPP=cpp
CFLAG=-Wall `pkg-config --cflags gtk+-2.0`
LDFLAG=`pkg-config --libs gtk+-2.0`
.PHONY: all run etags clean bak dep 

SRCS= $(wildcard *.c)
OBJS= $(patsubst %.c,%.o,$(SRCS))

all: run
$(PRJ): $(OBJS)
	$(CC) -o $@ $(LDFLAG) $^
%.o: %.c
	$(CC) -c  $(CFLAG) $<

run: $(PRJ)
	./$(PRJ)
etags:
	etags.emacs `find . -type f -name "*.[ch]"`
clean:
	rm -rf $(PRJ) *.o *~ .dep TAGS
bak: clean
	(cd ..; tar jcf $(PRJ).tar.bz2 $(PRJ); mv $(PRJ).tar.bz2 ~/arc)
dep:
	(for f in $(SRCS);do echo -n "";$(CPP) -M -MM -MG $$f;done) >> .dep

### Dependencies:
-include .dep
