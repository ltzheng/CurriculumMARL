#!/bin/bash

# ulimit -Sn unlimited && ulimit -Sl unlimited

ulimit -u 10240

pip install sklearn

algo=$1
teacher=$2

run_task() {
  if [[ "$teacher" == "none" ]]
  then
    for i in {1..2}; do
      # mod 4 since there 4 gpus
      gpuid=$(($i % 4))
      seed=$RANDOM
      exp_name=${algo}_${teacher}
      echo "Run ${exp_name}"
      CUDA_VISIBLE_DEVICES=${gpuid} setsid python /home/rundong/football-invariant_att_com/run_${algo}.py \
        --config-file=/home/rundong/football-invariant_att_com/configs/${algo}.yaml \
        --exp-name=${exp_name}\
        --seed=${seed} >/home/rundong/football-invariant_att_com/log/${exp_name}_${seed}.log 2>&1 &
      sleep 5
    done
  else
    for i in {1..2}; do
      # mod 4 since there 4 gpus
      gpuid=$(($i % 4))
      seed=$RANDOM
      exp_name=${algo}_${teacher}
      echo "Run ${exp_name}"
      CUDA_VISIBLE_DEVICES=${gpuid} setsid python /home/rundong/football-invariant_att_com/run_${algo}.py \
        --config-file=/home/rundong/football-invariant_att_com/configs/${algo}.yaml \
        --exp-name=${exp_name}\
        --teacher=${teacher}\
        --seed=${seed} >/home/rundong/football-invariant_att_com/log/${exp_name}_${seed}.log 2>&1 &
      sleep 5
    done
  fi
}

echo "run tasks"
run_task
wait
