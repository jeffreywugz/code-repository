target = list-based-queue
cc = gcc
cxx = g++
cflags = -lpthread -std=c99 -g
cxxflags = -lpthread -g

all: $(target).run
%.run: %.exe
	./$< $($*.args)
%.exe: %.c
	$(cc) $(cflags) -o $@ $<
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $<
notify.args = 24
read-write-lock.args = 20
seq-lock.args = 24
ticket-lock.args = 24
queue.args = 20
slice-pool.args = 20
rcu-lock.args = 20
list-based-queue.args = 20
