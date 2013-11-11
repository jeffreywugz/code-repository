disk-latency.args = 10000 10240
io-stress.args = async
cc = gcc
cxx = g++
cflags = -lpthread -lrt -std=c99 -g -O2 -D_GNU_SOURCE -laio
cxxflags = -lpthread -lrt -laio -g -O2 -D_GNU_SOURCE
sshpass=~/work/bin/sshpass -f root.pass
#remote-host=root@my163071.cm6
remote-host=root@10.13.249.192
remote-dir=/data/log1
all: dio-check.exe
help:
	vfiles.py *.png
io-stress.log: io-stress.exe
	path=tmp time_limit=10000000 write_size=1024 file_size=4096 ./$< $(io-stress.args)
disk-latency.log: disk-latency.exe
	time_limit=5000000 ./disk-latency.exe $(disk-latency.args)
rsync:
	$(sshpass) rsync -avz io-stress.exe run-io-stress.sh disk-latency.exe run-blktrace.sh run-disk-latency.sh $(remote-host):$(remote-dir)
# disk-latency.log blktrace.sdb.log: disk-latency.exe rsync
# 	$(sshpass) ssh $(remote-host) 'cd $(remote-dir); ./run-blktrace.sh'
# 	$(sshpass) rsync -av $(remote-host):$(remote-dir)/disk-latency.log .
# 	$(sshpass) rsync -av $(remote-host):$(remote-dir)/blktrace.sdb1.log blktrace.sdb.log
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
io-stress.exe: io-stress.cpp
