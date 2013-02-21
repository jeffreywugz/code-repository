disk-latency.args = 10000 10240
cc = gcc
cflags = -lpthread -lrt -std=c99 -g -O2 -D_GNU_SOURCE
sshpass=~/work/bin/sshpass -f root.pass
#remote-host=root@my163071.cm6
remote-host=root@172.24.131.193
remote-dir=/data/log1
all: blktrace.sda.png blktrace.sdb.png
help:
	vfiles.py *.png
rsync:
	$(sshpass) rsync -avz disk-latency.exe run-blktrace.sh $(remote-host):$(remote-dir)
disk-latency.log blktrace.sda.log blktrace.sdb.log: disk-latency.exe rsync
	#$(sshpass) ssh $(remote-host) 'cd $(remote-dir); ./run-blktrace.sh'
	$(sshpass) rsync -av $(remote-host):$(remote-dir)/disk-latency.log .
	$(sshpass) rsync -av $(remote-host):$(remote-dir)/blktrace.sda.log .
	$(sshpass) rsync -av $(remote-host):$(remote-dir)/blktrace.sdb.log .
disk-latency.tab: disk-latency.log
	~/ob.dev/tools/deploy/b/findall.py ':$${latency:int:\d+}$$' < $< > $@
blktrace.%.tab: blktrace.%.log
	./blktrace-sort.py $< > $@
blkwrite.%.tab: blktrace.%.log
	./blktrace-writenum.py $< >$@
disk-latency.png: disk-latency.tab
	rm -r *.db
	~/ob.dev/tools/deploy/b/tquery.py '$<' 'select plot("$@,b,", latency) from t_'
blktrace.%.png: blktrace.%.tab
	~/ob.dev/tools/deploy/b/tquery.py '$<' 'select plot("$@,b,", ts, duration) from t_ where ev="D-C" order by ts'
%.run: %.exe
	./$< $($*.args)
%.exe: %.c
	$(cc) $(cflags) -o $@ $<
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $<