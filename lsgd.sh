#!/bin/bash
DATE=`date +%Y-%m-%d`

ip_addr='192.168.238.145' # user-defined IP Address
cur_group=$1
num_groups=4 # total number of GPU nodes

lr=0.1
c1=0.1
c2=0.1
l_comm=64
g_comm=128

echo "STARTING LSGD-lr=$lr(l_comm=$l_comm, g_comm=$g_comm)"
python ./codes/main.py --distributed --l_comm $l_comm --g_comm $g_comm \
--dist_optimizer lsgd --datadir './dataset' --dataset 'cifar' \
--batch_size 128 --lr $lr --c1 $c1 --c2 $c2 --mom 0 \
--dist_ip $ip_addr --dist_port 2432 --num_groups $num_groups --cur_group $cur_group \
--exp_name "LSGD-lr-$lr-l_comm=$l_comm=g_comm=$g_comm=c1-$c1-c2-$c2" \
--checkpoints_dir "LSGD-cifar-w$num_swarms-$cur_swarm-$DATE"  \
--check_point_epochs 50  --minutes 10
echo "FINISHING LSGD-lr=$lr(l_comm=$l_comm, g_comm=$g_comm)"
pkill -9 python
