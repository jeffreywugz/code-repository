PRJ=svm-app
.PHONY: all run clean bak

all: run

libsvm-2.89/python/svmc.so:
	make -C libsvm-2.89/python
run: libsvm-2.89/python/svmc.so
	./$(PRJ).py
clean:
	rm -rf *.~ *.pyc
	make clean -C libsvm-2.89/python
bak: clean
	(cd ..; tar jcf $(PRJ).tar.bz2 $(PRJ); mv $(PRJ).tar.bz2 ~/arc)
