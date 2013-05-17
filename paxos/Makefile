target=ps
cc=gcc
cpp=cpp
cflags=-Wall -Werror -g
ldflags=
.PHONY: all run etags clean dep bak

srcs=$(wildcard *.c)
objs=$(patsubst %.c,%.o,$(srcs))

all: $(target) run
$(target): $(objs)
	$(cc) -o $@ $(ldflags) $^
%.o: %.c
	$(cc) -c  $(cflags) $<
run: $(target)
	./$(target)
etags:
	etags.emacs `find . -type f -name "*.[ch]"`
clean:
	rm -rf $(target) *.o *~ .dep
realclean: clean
	rm -rf TAGS
bak: clean
	(cd ..; tar jcf $(target).tar.bz2 $(target); mv $(target).tar.bz2 ~/arc)
dep:
	(for f in $(src);do echo -n "";$(cpp) -M -MM -MG $$f;done) >> .dep

### Dependencies:
-include .dep
