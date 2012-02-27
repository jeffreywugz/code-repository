target = test
TARGET = $(shell echo $(target) | tr 'a-z' 'A-Z')
cc = gcc
cxx = g++
cflags = -std=c99 -D__TEST_$(TARGET)__
cxxflags = -D__TEST_$(TARGET)__
ldflags = -lpthread
all: $(target).test
%.test: %.exe
	./$<
%.exe:
	$(cxx) $(ldflags) -o $@ $^
%.o: %.c
	$(cc) $(cflags) -c -o $@ $^
%.o: %.cpp
	$(cxx) $(cxxflags) -c -o $@ $^
srcs= $(wildcard *.cpp)
objs= $(patsubst %.cpp,%.o,$(srcs))
$(target).exe: $(objs)
clean:
	rm *.o *.exe