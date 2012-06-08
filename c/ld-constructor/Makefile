target = so
args = test
cc = gcc
cflags =
ldflags = -ldl -lpthread

all: so.test
%.test: %
	./$<
	LD_PRELOAD=./$< ls
$(target): $(target).c
	$(cc) -shared -fPIC -Wl,-e,mystart -o $@ $^  $(cflags) $(ldflags)
a: a.c
	$(cc) -o $@ $<
clean:
	rm -f *.so *.o