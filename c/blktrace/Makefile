disk-latency.args = 10000 10240
cc = gcc
cflags = -lpthread -lrt -std=c99 -g -O2 -D_GNU_SOURCE
sshpass=~/work/bin/sshpass -f root2.pass
remote-host=root@my163071.cm6
all: blktrace.png
rsync:
	$(sshpass) rsync -avz disk-latency.exe run-blktrace.sh $(remote-host):
disk-latency.log blktrace.log: disk-latency.exe rsync
	#$(sshpass) ssh $(remote-host) ./run-blktrace.sh
	$(sshpass) rsync -av $(remote-host):disk-latency.log .
	$(sshpass) rsync -av $(remote-host):blktrace.log .
disk-latency.tab: disk-latency.log
	~/ob.dev/tools/deploy/b/findall.py ':$${latency:int:\d+}$$' < $< > $@
blktrace.tab: blktrace.log
	./blktrace-sort.py $< > $@
disk-latency.png: disk-latency.tab
	rm -r *.db
	~/ob.dev/tools/deploy/b/tquery.py '$<' 'select plot("$@,b,", latency) from t_'
blktrace.png: blktrace.tab
	~/ob.dev/tools/deploy/b/tquery.py '$<' 'select plot("$@,b,", ts, duration) from t_ where ev="D-C" order by ts'
%.run: %.exe
	./$< $($*.args)
%.exe: %.c
	$(cc) $(cflags) -o $@ $<
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $<
