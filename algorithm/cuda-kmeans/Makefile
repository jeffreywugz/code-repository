PRJ=kmeans
CC=g++
NVCC=nvcc
CPP=cpp
CFLAG=-Wall
#NVCFLAG=--device-emulation
# LDFLAG=-g -pg
LDFLAG=
.PHONY: all run etags clean bak dep 

SRCS= $(wildcard *.cpp)
CUSRCS= $(wildcard *.cu)
OBJS= $(patsubst %.cpp,%.o,$(SRCS)) $(patsubst %.cu,%.o,$(CUSRCS))

all: run
$(PRJ): $(OBJS)
	$(NVCC) $(NVCFLAG) -o $@ $(LDFLAG) $^
%.o: %.cpp
	$(CC) -c  $(CFLAG) $<
%.o: %.cu
	$(NVCC) -c  $(NVCFLAG) $<
run: $(PRJ)
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
