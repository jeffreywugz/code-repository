prj=jigsaw

.PHONY:run clean
all: run
jigsaw:jigsaw.cpp
	g++ -o $@ $^
run:jigsaw
	./ui.py
clean:
	rm -rf jigsaw *.pyc
