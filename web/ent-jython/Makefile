class-path = $$CLASSPATH:jar/vijava2u120091204.jar:jar/dom4j-1.6.1.jar:jar/jna.jar:jar/vix.jar
javacflags = -classpath $(class-path)
javaflags = -classpath $(class-path)
mxmlc = mxmlc
mxmlcflags = -use-network  --library-path+=. --show-actionscript-warnings=true --strict=true --verbose-stacktraces=true

.PHONY: all run test clean browse
all: clean-log ent.swf test-agent
agent-eval = ./agent.py $(1)
ent-eval = ./agent.py raw $(1)
test-vc:
	$(call ent-eval,vc.test 10.117.5.68 probe.2)
bootstrap:
	sshpass -p iambook11 scp -r probevm root@10.117.5.120:/vmfs/volumes/Storage1/
register-plugin:
	$(call ent-eval,ent.register http://10.117.4.52:8000/plugin.xml)
test-ent:
	$(call ent-eval,test 10.117.5.68 10.117.5.120)
	# $(call ent-eval,getHosts)
	# $(call ent-eval,ent.viewResult)
	# $(call ent-eval,ent.probeVms.freeAll)
	# $(call ent-eval,ent.check 10.117.5.68 test 10.117.5.68 'VM Network')
	# $(call ent-eval,ent.probeVms.cleanup)
test-agent:
	# $(call agent-eval,getHosts)
	$(call agent-eval,getAllPortgroups)
	# $(call agent-eval,check 10.117.5.68 test 10.117.5.68 'VM Network')

restart-server:
	pkill -9 -f -x './server' || true
	./server.py
%.swf: %.mxml
	$(mxmlc) $(mxmlcflags) $<
ent.swf: ent.mxml css/flex.css mflex ent

clean-log:
	rm log/* -rf
interactive:
	LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:./lib  CLASSPATH=$(class-path) jython -i ./$(prj).py
%.jar: %
	javac $(javacflags) $</*.java
	jar cvfm $@ $</manifest.mf $<
browse: snippet/vijavaBrowser.jar
	java $(javaflags) -classpath $< vijavaBrowser.Application
clean:
	rm -rf *.pyc *.swf *.class */*.class