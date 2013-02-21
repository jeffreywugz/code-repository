dev="sda sdb"
data_dir=/home/admin/tmp
echo 'kill blktrace and ./disk-latency.exe'
pkill blktrace
pkill -f ./disk-latency.exe
echo $PATH
rm sda.blktrace* sdb.blktrace* -f
echo 'start blktrace'
blktrace /dev/sda /dev/sdb &
echo 'start disk-latency.exe'
mkdir -p $data_dir
file_size=10 time_limit=3600000000 path=$data_dir ./disk-latency.exe 10000 10240 > disk-latency.log
echo 'kill blktrace'
pkill blktrace
echo 'run blkparse'
blkparse sda > blktrace.sda.log
blkparse sdb > blktrace.sdb.log
