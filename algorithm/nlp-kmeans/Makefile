PRJ=kmeans
CC=g++
CPP=cpp
CFLAG=-Wall -O3
LDFLAG=-g -O3
.PHONY: all run etags clean bak dep

SRCS= $(wildcard *.cpp)
OBJS= $(patsubst %.cpp,%.o,$(SRCS))
include config.txt
all: run
$(PRJ): $(OBJS)
	$(CC) $(CFLAG) -o $@ $(LDFLAG) $^
$(corpus_file):
	./main.py
%.o: %.cpp
	$(CC) -c  $(CFLAG) $<
run: $(PRJ) $(corpus_file)
	./$(PRJ) config.txt
etags:
	etags.emacs `find . -type f -name "*.[ch]"`
clean:
	rm -rf $(PRJ) *.o *~ .dep TAGS
bak: clean
	(cd ..; tar jcf $(PRJ).tar.bz2 $(PRJ); mv $(PRJ).tar.bz2 ~/arc)
dep:
	(for f in $(SRCS) $(CUSRCS);do echo -n "";$(CPP) -M -MM -MG $$f;done) >> .dep

### Dependencies:
-include .dep
