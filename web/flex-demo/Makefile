prj = hadoop
mxmlc = mxmlc
# mxmlc = ./fcsh-wrap
mxmlcflags = -use-network  --library-path+=. --show-actionscript-warnings=true --strict=true --verbose-stacktraces=true

.PHONY: all start-server remake clean
all: $(prj).swf

start-server:
	./server.py
%.swf: %.mxml
	$(mxmlc) $(mxmlcflags) $<
hadoop.swf: hadoop.mxml css/flex.css mflex hadoop
clean:
	rm -rf *.pyc *.swf examples/*.swf
