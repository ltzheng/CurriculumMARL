#!/bin/bash -l


# ```non-hrl``` ```hrl``` ```shared``` + ```two teacher``` (3 * 2)
declare -a algos=("gfootball_shared_ppo_curriculum") #["gfootball_att_com_curriculum_hrl" "gfootball_att_com_curriculum" "gfootball_shared_ppo_curriculum"]
declare -a teachers=("time_teacher") #("time_teacher" "discrete_bandit" "simple_bandit" "task_wise_bandit" "eval_on_5v5_bandit")

# qmix shared ppo (2 * 1)
#declare -a algos=("gfootball_qmix" "gfootball_shared_ppo") #["gfootball_shared_ppo", "shared_curriculum", "com", "com_curriculum" "gfootball_att_com_curriculum_hrl" "gfootball_qmix"]
#declare -a teachers=("none") #("discrete_bandit" "simple_bandit" "task_wise_bandit" "eval_on_5v5_bandit")

for algo in ${algos[@]}; do
  for teacher in ${teachers[@]}; do
      sbatch football-invariant_att_com/run_sbatch.sh ${algo} ${teacher}
  done
done