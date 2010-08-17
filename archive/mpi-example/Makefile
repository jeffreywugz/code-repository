PRJ=matmul
CC=mpicc
CPP=cpp
CFLAGS=-Wall -g -lm
.PHONY: all run etags clean bak dep 

all: run
$(PRJ): $(PRJ).c
	$(CC) -o $@ $(CFLAGS) $<

run: $(PRJ)
	mpiexec -n 2 ./$(PRJ)
etags:
	etags.emacs `find . -type f -name "*.[ch]"`
clean:
	rm -rf $(PRJ) *.o *~ .dep TAGS
bak: clean
	(cd ..; tar jcf $(PRJ).tar.bz2 $(PRJ); mv $(PRJ).tar.bz2 ~/arc)
