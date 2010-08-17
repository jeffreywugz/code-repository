prj = counter
cc = gcc
cflags = -Wall -Werror -g
ldflags = -lpthread
.PHONY: all run clean
all: run
$(prj): $(prj).o
	$(cc) $(ldflags) -o $@ $<
%.o: %.c
	$(cc) $(cflags) -c -o $@ $<
run: $(prj)
	taskset 0x00000001 ./$(prj)
	# taskset 0x00000003 ./$(prj)
	# taskset 0x0000000f ./$(prj)
	taskset 0xffffffff ./$(prj)
clean:
	rm -rf *.pyc *.o $(prj)