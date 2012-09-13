target = glob
all: test
test: $(target).exe
	./$<
%.exe: %.c
	gcc -o $@ $<
