disk-latency.args = 10000 10240
cc = gcc
cflags = -lpthread -lrt -std=c99 -g -O2 -D_GNU_SOURCE
sshpass=~/work/bin/sshpass -f root.pass
#remote-host=root@my163071.cm6
remote-host=root@172.24.131.193
remote-dir=/data/log1
all: disk-latency.png
rsync:
	$(sshpass) rsync -avz disk-latency.exe $(remote-host):$(remote-dir)
disk-latency.log blktrace.sda.log blktrace.sdb.log: disk-latency.exe rsync
	$(sshpass) ssh $(remote-host) 'cd $(remote-dir); time_limit=30000000 write_ahead_size=512 ./disk-latency.exe 100 10240'
	$(sshpass) rsync -av $(remote-host):$(remote-dir)/disk-latency.log .
disk-latency.tab: disk-latency.log
	~/ob.dev/tools/deploy/b/findall.py ':$${latency:int:\d+}$$' < $< > $@
disk-latency.png: disk-latency.tab
	rm -r *.db
	~/ob.dev/tools/deploy/b/tquery.py '$<' 'select plot("$@,b,", latency) from t_'
%.run: %.exe
	./$< $($*.args)
%.exe: %.c
	$(cc) $(cflags) -o $@ $<
%.exe: %.cpp
	$(cxx) $(cxxflags) -o $@ $<