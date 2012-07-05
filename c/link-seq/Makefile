target = link-using-obj-file link-using-static-lib-wrong-order link-using-static-lib-correct-order
cc = gcc
all: $(target)
link-using-obj-file: link-seq.c bar.o foo.o
	$(cc) -o $@ $^
link-using-static-lib-wrong-order: link-seq.c bar.a foo.a
	$(cc) -o $@ $^
link-using-static-lib-correct-order: link-seq.c foo.a bar.a
	$(cc) -o $@ $^
%.o: %.c
	$(cc) -o $@ -c $^
%.a: %.o
	ar r $@ $^