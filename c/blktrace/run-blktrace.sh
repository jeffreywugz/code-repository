dev=sda12
data_dir=/u02/tmp
echo 'kill blktrace and ./disk-latency.exe'
pkill blktrace
pkill -f ./disk-latency.exe
echo $PATH
rm $dev.blktrace* -f
echo 'start blktrace'
blktrace /dev/$dev &
echo 'start disk-latency.exe'
mkdir -p $data_dir
file_size=10 time_limit=600000000 path=$data_dir ./disk-latency.exe 10000 10240 > disk-latency.log
echo 'kill blktrace'
pkill blktrace
echo 'run blkparse'
blkparse $dev > blktrace.log
