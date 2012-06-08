target = alloca
type = alloca
cc = gcc
all: $(target).test
%.test: %
	pkill -f '^./$<' || true
	./$< $(type) 10240000 &
	cat /proc/`pgrep $<`/maps
alloca: alloca.c
	$(cc) -o $@ $<