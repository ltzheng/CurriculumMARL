#!/bin/bash -l
#SBATCH --account=p200009                  # project account
#SBATCH --partition=gpu                    # partition
#SBATCH --qos=default                        # QOS default, short, urgent
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=1                         # number of tasks
#SBATCH --ntasks-per-node=1                # number of tasks per node
#SBATCH --cpus-per-task=128                # number of cores per task
#SBATCH --time=01-00:00                     # time (DD-HH:MM)

algo=$1
teacher=$2

echo "Run module"
module load Singularity-CE/3.8.4
singularity instance start --nv -B /project/home/p200009/rundong:/home/rundong football.simg football
singularity exec -H /project/home/p200009/rundong:/home/rundong instance://football bash /home/rundong/football-invariant_att_com/run_football_in_singularity.sh ${algo} ${teacher}