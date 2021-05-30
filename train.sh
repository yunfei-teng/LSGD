#!/bin/bash 

# -------- Experiment Setup --------
DATE=`date +%Y-%m-%d`
dataset='cifar'; model='resnet20'; datadir='./dataset'
m=0.9; batch_size=128; minutes=12

# -------- Required Info --------
num_groups={'number of groups'}
if [[ $num_groups -eq 1 ]]
then
  cur_group=0; ip_addr={'ip address of current node'}
else
  cur_group=$1; ip_addr={'ip address of the first node'}
fi

# -------- Optimizer: LSGD --------
dist_op='LSGD'
check_dir="LSGD-$dataset-$model-w$num_groups-$cur_group-$DATE"
c1=0.1; c2=0.1; p1=0.1; p2=0.1
for g_comm in 4 16 64; do
  for lr in 1e-1 1e-2 1e-3; do
    l_comm=$(expr $g_comm / 4); avg_size=$l_comm
    exp_name="$dist_op-lr-$lr-l_comm=$l_comm=g_comm=$g_comm=c1-$c1-c2-$c2-p1-$p1-p2-$p2-m-$m-a-$avg_size="

    echo "STARTING LSGD-lr=$lr(l_comm=$l_comm, g_comm=$g_comm)"
    python ../codes/main.py --distributed --l_comm $l_comm --g_comm $g_comm \
      --dist_optimizer $dist_op --datadir $datadir --dataset $dataset --model $model \
      --batch_size $batch_size --lr $lr --c1 $c1 --c2 $c2 --p1 $p1 --p2 $p2 --mom $m --avg_size $avg_size \
      --dist_ip $ip_addr --dist_port 2432 --num_groups $num_groups --cur_group $cur_group \
      --exp_name $exp_name --checkpoints_dir $check_dir  --minutes $minutes
    echo "FINISHING LSGD-lr=$lr(l_comm=$l_comm, g_comm=$g_comm)"
  done
done

# -------- Optimizer: EASGD --------
dist_op='EASGD'
check_dir="EASGD-$dataset-$model-w$num_groups-$cur_group-$DATE"
c1=0.0; c2=0.43; p1=0.0; p2=0.0
for g_comm in 4 16 64; do
    for lr in 1e-1 1e-2 1e-3; do
    l_comm=$(expr $g_comm)
    exp_name="$dist_op-lr-$lr-comm=$g_comm=c2-$c2-m-$m"
    echo "STARTING $dist_op-lr=$lr(comm=$g_comm)"
    python ../codes/main.py --distributed --l_comm $l_comm --g_comm $g_comm \
      --dist_optimizer $dist_op --datadir $datadir --dataset $dataset --model $model \
      --batch_size $batch_size --lr $lr --c1 $c1 --c2 $c2 --p1 $p1 --p2 $p2 --mom $m \
      --dist_ip $ip_addr --dist_port 2432 --num_groups $num_groups --cur_group $cur_group \
      --exp_name $exp_name --checkpoints_dir $check_dir  --minutes $minutes
    echo "FINISHING $dist_op-lr=$lr(comm=$g_comm)"
  done
done

# -------- Optimizer: Data Parallel --------
dist_op='DataParallel'
check_dir="DataParallel-$dataset-$model-w$num_groups-$cur_group-$DATE"
g_comm=1
for lr in 1e-1 1e-2 1e-3; do
    exp_name="DataParallel-lr-$lr-m-$m-b-$batch_size-comm=$g_comm="
    echo "STARTING DataParallel-lr=$lr(comm=$g_comm)"
    python ../codes/data_parallel.py --datadir $datadir --dataset $dataset --model $model \
      --optimizer 'SGD' --batch_size $batch_size --lr $lr --mom $m --dist_ip $ip_addr --num_groups $num_groups --cur_group $cur_group \
      --exp_name $exp_name --checkpoints_dir $check_dir --minutes $minutes
    echo "FINISHING DataParallel-lr=$lr(comm=$g_comm)" 
done

# -------- Optimizer: LARS --------
dist_op='LARS'
check_dir="LARS-$dataset-$model-w$num_groups-$cur_group-$DATE"
g_comm=1
for lr in 1e2 1e1 1e0 1e-1 1e-2; do
    exp_name="LARS-lr-$lr-m-$m-b-$batch_size-comm=$g_comm="
    echo "STARTING LARS-lr=$lr(comm=$g_comm)"
    python ../codes/data_parallel.py --datadir $datadir --dataset $dataset --model $model \
    --optimizer 'LARS' --batch_size $batch_size --lr $lr --mom $m --dist_ip $ip_addr --num_groups $num_groups --cur_group $cur_group \
    --exp_name $exp_name --checkpoints_dir $check_dir --minutes $minutes
    echo "FINISHING LARS-lr=$lr(comm=$g_comm)" 
done