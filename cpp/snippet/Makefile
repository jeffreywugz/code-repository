target = tsi
cxx = g++
cxxflags = -lpthread -g -O3

all: $(target).run
%.run: %.exe
	./$< $($*.args)
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $<
tsi.args =
