queue_conf=$PWD/slurm_conf/slurm.conf
#export train_cmd="slurm.pl --config $queue_conf"
export train_cmd="run.pl"
export cuda_cmd="run.pl"
export slurm_cmd="slurm.pl --config $queue_conf"
