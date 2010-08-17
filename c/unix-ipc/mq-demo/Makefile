PRJ=mq-demo
CC=gcc
CPP=cpp
CFLAG=-Wall -Werror -g
LDFLAG=-lrt
.PHONY: all run check-syntax etags clean realclean bak dep 

SRCS= $(wildcard *.c)
OBJS= $(patsubst %.c,%.o,$(SRCS))

all: run
$(PRJ): $(OBJS)
	$(CC) -o $@ $(LDFLAG) $^
%.o: %.c
	$(CC) -c  $(CFLAG) $<

check-syntax:
	$(CC) -Wall -fsyntax-only $(CHK_SOURCES)
run: $(PRJ)
	./$(PRJ)
etags:
	etags.emacs `find . -type f -name "*.[ch]"`
clean:
	rm -rf $(PRJ) *.o *~ .dep
realclean: clean
	rm -rf TAGS
bak: clean
	(cd ..; tar jcf $(PRJ).tar.bz2 $(PRJ); mv $(PRJ).tar.bz2 ~/arc)
dep:
	(for f in $(SRCS);do echo -n "";$(CPP) -M -MM -MG $$f;done) >> .dep

### Dependencies:
-include .dep
