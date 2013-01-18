target = epoll
disk-latency.args = 10000 10240
new-queue.args = 1 1 10000
spin-queue.args = 1 1 100000000
queue-with-sem.args = 1 1 10000000
rol.args =
queue.args = 2 10000000
profile-frame.args = 4 13
sem-vs-pthread-signal.args = 24 1
cas-vs-atomic-add.args = 2 100000000
fadvise-seq.args = 10240000000
memcpy.args = 1024000000
memcmp.args =
mem-param.args = 1024 8 1000000
gcc-opt.args =
cc = gcc
cxx = g++
cflags = -lpthread -lrt -std=c99 -g -O2 -D_GNU_SOURCE
cxxflags = -g -O2 -lpthread -lrt

all: blktrace.png
disk-latency.report: remote-host=172.24.131.194
disk-latency.report blktrace.log: disk-latency.exe
	rsync -avz disk-latency.exe $(remote-host):
	ssh $(remote-host) file_size=10 time_limit=3000000000 path=/data/log1/tmp ./$< $(disk-latency.args) >$@
disk-latency.tab: disk-latency.report
	~/ob.dev/tools/deploy/b/findall.py ':$${latency:int:\d+}$$' < $< > $@
blktrace.tab: blktrace.log
	./blktrace-sort.py $< > $@
blktrace.png: blktrace.tab
	~/ob.dev/tools/deploy/b/tquery.py '$<' 'select plot("$@,b,", ts, duration) from t_ where ev="D-C" order by ts'
pkt_size=64
net-latency.report: host1=172.24.131.193
net-latency.report: host2=172.24.131.194
net-latency.report: epoll.exe
	rsync -avz $< $(host1): && rsync -avz $< $(host2):
	ssh $(host1) "pkt_size=$(pkt_size) ./epoll.exe server $(host1) 9000 > server.log &"
	ssh $(host2) duration=2000000000 client_wait_us=1000 pkt_size=$(pkt_size) ./epoll.exe client $(host1) 9000 1 > $@
net-latency.tab: net-latency.report
	~/ob.dev/tools/deploy/b/findall.py ':$${time1:int:\d+}:$${time2:int:\d+}$$' < $< > $@
%.run: %.exe
	./$< $($*.args)
%.exe: %.c
	$(cc) $(cflags) -o $@ $<
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $<
