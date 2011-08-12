exe = hello
trace = ./trace.py
all: $(exe).trace
%.plain: %.trace %.exe
	$(reduce) $*.exe plain <$< >$@
%.dot: %.trace %.exe
	$(reduce) $*.exe dot <$< >$@
%.png: %.dot
	dot -Tpng $< -o $@

$(exe).trace: $(exe).exe $(trace-lib).so
	TRACE=$(exe).trace LD_PRELOAD=./$(trace-lib).so ./$(exe).exe
$(exe).png: $(exe).exe $(trace)
	TRACE=$(exe).png ./$(trace) ./$(exe).exe
%.so: %.c
	gcc -fPIC --shared -rdynamic -ldw -o $@ $<
%.exe: %.c
	gcc -finstrument-functions -g -o $@ $<
%.exe: %.cpp
	g++ -finstrument-functions -g -o $@ $<
clean:
	rm -rf *.so *.exe *.trace *.dot *.plain *.png