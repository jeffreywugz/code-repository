cc = gcc
cflags = -g
ldflags = -g
target = python/lock/semaphore.py
all: test
test: $(target)
	(time ./$(target) )
%.exe: %.o
	$(cc) $(ldflags) -o $@ $<
%.o: %.c
	$(cc) $(ldflags) -o $@ -c $<
clean:
	rm -rf $(target)