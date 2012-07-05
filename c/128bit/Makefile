all: test

test: speed-test
	./$< 28
speed-test: speed-test.c
	gcc --std=c99 -o $@ $<
128bit: 128bit.c
	gcc -o $@ $<
