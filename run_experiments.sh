#!/bin/bash
SEED=$RANDOM
teachers=("bandit" "ppo" "alp_gmm" "uniform" "vacl")
for i in ${!teachers[@]}; do
 GPU=$(($i % 4))
 NAME=${USER}_football_GPU_${GPU}_${SEED}_${teachers[$i]}
 echo "Launching container '${NAME}' on GPU '${GPU}'"
 # Launches a docker container using our image, and runs the provided command
 NV_GPU="${GPU}" nvidia-docker run \
     -d \
     --rm \
     --name ${NAME} \
     -v `pwd`:/home/football \
     spc \
     python train.py -f configs/football/${teachers[$i]}/corner.yaml --seed ${SEED}
 sleep 5
done


#!/bin/bash
#for i in {1..5}; do
#  SEED=$RANDOM
#  GPU=$(($i % 4))
#  NAME=${USER}_football_GPU_${GPU}_${SEED}_bandit
#  echo "Launching container '${NAME}' on GPU '${GPU}'"
#  NV_GPU="${GPU}" nvidia-docker run \
#      -d \
#      --rm \
#      --name ${NAME} \
#      -v `pwd`:/home/football \
#      spc \
#      python train.py -f configs/football/bandit/corner.yaml --seed ${SEED}
#  sleep 5
#done


# nvidia-docker run -it --rm --name test -v `pwd`:/home/football spc \
#       python train.py -f configs/football/bandit/corner.yaml