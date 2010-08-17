PRJ=doxygen-demo
CC=gcc
CPP=cpp
CFLAG=-Wall -Werror -g
LDFLAG=
DOXYGEN.CONF=doxygen.conf
.PHONY: all run doc  etags clean realclean bak dep 

SRCS= $(wildcard *.c)
OBJS= $(patsubst %.c,%.o,$(SRCS))

all: $(PRJ) 
$(PRJ): $(OBJS)
	$(CC) -o $@ $(LDFLAG) $^
%.o: %.c
	$(CC) -c  $(CFLAG) $<

doc:
	doxygen $(DOXYGEN.CONF)
run: $(PRJ)
	./$(PRJ)
etags:
	etags.emacs `find . -type f -name "*.[ch]"`
clean:
	rm -rf $(PRJ) *.o *~ .dep html
realclean: clean
	rm -rf TAGS
bak: clean
	(cd ..; tar jcf $(PRJ).tar.bz2 $(PRJ); mv $(PRJ).tar.bz2 ~/arc)
dep:
	(for f in $(SRCS);do echo -n "";$(CPP) -M -MM -MG $$f;done) >> .dep

### Dependencies:
-include .dep
