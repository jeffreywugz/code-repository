PRJ=graph
.PHONY:dsp clean

dsp:$(PRJ).py
	#python setup.py install --prefix=~/local
	python setup.py build_ext --inplace
	./$(PRJ).py input output.mp
	mpost output.mp
	evince output.1

clean:
	rm -rf *~ *.pyc *.mp output.log *.1 *so build

