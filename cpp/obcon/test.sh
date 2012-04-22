
#export log_file=test.log
app=./client.exe
rs=$1
ups=$2
set -x
$app desc $rs
$app get_obi_role $rs
$app get_master_ups $rs
$app get_client_cfg $rs
$app get_ms $rs
$app get_ms $rs
$app get_replayed_cursor $ups
$app get_max_log_seq_replayable $ups
$app get_last_frozen_version $ups
$app set $rs table1 column_2 11223344556677889911223344556677881122334455667788991122334455667788 'setbyme'
$app scan $rs table1








