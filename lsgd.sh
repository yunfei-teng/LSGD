#!/bin/bash
DATE=`date +%Y-%m-%d`

cur_group=$1
num_groups=['Number of Nodes']
ip_addr=['IP address of an anbitrary node']

l_comm=64
g_comm=128

lr=['Learning Rate']
c1=['Lambda']
c2=['Lambda_G']

echo "STARTING LSGD-lr=$lr(l_comm=$l_comm, g_comm=$g_comm)"
python ./codes/main.py --distributed --l_comm $l_comm --g_comm $g_comm \
--dist_optimizer lsgd --datadir './dataset' --dataset 'cifar' \
--batch_size 128 --lr $lr --c1 $c1 --c2 $c2 --mom 0 \
--dist_ip $ip_addr --dist_port 2432 --num_groups $num_groups --cur_group $cur_group \
--exp_name "LSGD-lr-$lr-l_comm=$l_comm=g_comm=$g_comm=c1-$c1-c2-$c2" \
--checkpoints_dir "LSGD-cifar-$num_groups-$cur_group-$DATE" \
--check_point_epochs 50  --minutes 10
echo "FINISHING LSGD-lr=$lr(l_comm=$l_comm, g_comm=$g_comm)"

pkill -9 python
