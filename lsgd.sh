#!/bin/bash
ip_addr='' # user-defined IP Address
num_swarms= # total number of GPU nodes
cur_swarm=$1
DATE=`date +%Y-%m-%d`
c1=0.1
c2=0.1
g_comm=128
l_comm=64
lr=0.1

echo "STARTING SOSGD-lr=$lr(l=$l_comm)"
python ./LSGD/main.py --distributed --l_comm $l_comm --dist_optimizer lsgd --datadir './dataset' \
--lr $lr --dist_ip $ip_addr --dist_port 2432 --dataset 'cifar' --exp_name "LSGD-lr-$lr-l_comm=$l_comm=g_comm=$g_comm=c1-$c1-c2-$c2" \
--checkpoints_dir "LSGD-cifar-w$num_swarms-$cur_swarm-$DATE" --batch_size 128 --c1 $c1 --c2 $c2 \
--num_swarms $num_swarms --cur_swarm $cur_swarm  --check_point_epochs 50 --mom 0 --minutes 10
echo "FINISHING SOSGD-lr=$lr(l=$l_comm)"
pkill -9 python
