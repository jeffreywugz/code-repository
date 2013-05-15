dev=sdb1
data_dir=/data/log1/tmp
echo 'kill blktrace and ./disk-latency.exe'
pkill blktrace
pkill -f ./disk-latency.exe
echo $PATH
rm sda.blktrace* sdb.blktrace* $dev.blktrace* -f
echo 'start blktrace'
blktrace -d /dev/$dev &
echo 'start disk-latency.exe'
mkdir -p $data_dir
file_size=10 time_limit=60000000 path=$data_dir ./disk-latency.exe 10000 10240 > disk-latency.log
echo 'kill blktrace'
pkill blktrace
echo 'run blkparse'
blkparse $dev > blktrace.$dev.log
