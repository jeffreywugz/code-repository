target=ps
cc=gcc
cpp=cpp
cflags=-Wall -Werror -g
ldflags=
.PHONY: all run etags clean dep bak

srcs=$(wildcard *.c)
objs=$(patsubst %.c,%.o,$(srcs)) $(target).src.o

all: $(target) run
$(target): $(objs)
	$(cc) -o $@ $(ldflags) $^
%.o: %.c
	$(cc) -c  $(cflags) $<
run: $(target)
	./$(target)
$(target).tar.bz2: $(srcs) Makefile
	tar jcf $@ $^
$(target).src.o: $(target).tar.bz2
	ld -r -b binary -o $@ $< # objcopy -B i386 -I binary -O elf64-x86-64 $< $@
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
