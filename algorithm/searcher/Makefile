PRJ=searcher
.PHONY: all run clean bak

all: run
run:
	./$(PRJ).py

clean:
	rm -rf *.pyc *~

bak: clean
	(cd ..; tar jcf $(PRJ).tar.bz2 $(PRJ); mv $(PRJ).tar.bz2 ~/arc)
