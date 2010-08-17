PRJ=cpio_parser
CC=gcc
CPP=cpp
CFLAG=-Wall -g
LDFLAG=
.PHONY: all run etags clean bak dep 

SRCS= $(wildcard *.c)
OBJS= $(patsubst %.c,%.o,$(SRCS))

all: run
$(PRJ): $(OBJS)
	$(CC) -o $@ $(LDFLAG) $^
%.o: %.c
	$(CC) -c  $(CFLAG) $<
a.cpio: sample
	find sample|cpio -H newc -o >a.cpio
run: $(PRJ)
	./$(PRJ) <a.cpio
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
