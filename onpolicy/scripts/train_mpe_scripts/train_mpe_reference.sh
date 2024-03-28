#!/bin/sh
env="MPE"
scenario="simple_reference"
num_landmarks=3
num_agents=2
algo="mappo" #"rmappo" "ippo"
exp="check"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 25 --num_env_steps 3000000 \
    --ppo_epoch 15 --gain 0.01 --lr 7e-4 --critic_lr 7e-4
done

# --env_name "MPE" --algorithm_name "rmappo" --experiment_name "check" --scenario_name "simple_reference" --num_agents 2 --num_landmarks 3 --seed 1 --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 25 --num_env_steps 3000000 --ppo_epoch 15 --gain 0.01 --lr 7e-4 --critic_lr 7e-4