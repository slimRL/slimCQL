#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@ --first_seed dummy --last_seed dummy
FIRST_SEED=$((N_PARALLEL_SEEDS * (SLURM_ARRAY_TASK_ID - 1) + 1)) 
LAST_SEED=$((N_PARALLEL_SEEDS * SLURM_ARRAY_TASK_ID))

source env_cpu/bin/activate

for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    python3 experiments/$ENV_NAME/$ALGO_NAME.py --experiment_name $EXPERIMENT_NAME --seed $seed $ARGS &> experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/seed_$seed.out & 
done
wait

N_PARALLEL_EPOCHS=4
N_EPOCHS=$(echo "$ARGS" | grep -oP '(?<=--n_epochs |-ne )\d+')

for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    epoch=0
    while (( epoch < N_EPOCHS+1 ))
    do
        python3 experiments/$ENV_NAME/evaluate.py --experiment_name $EXPERIMENT_NAME --algo_name $ALGO_NAME --seed $seed --epoch $epoch --horizon 100 --n_evaluation_steps_per_epoch 1000 >> experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/seed_$seed.out 2>&1 &
        
        ((epoch++))
        
        if (( epoch % N_PARALLEL_EPOCHS == 0 )); then
            wait
        fi
    done
    wait
    python3 experiments/synchronize_evaluation_wandb.py --experiment_name $EXPERIMENT_NAME --algo_name $ALGO_NAME --env_name "car_on_hill" --seed $seed --delete_models >> experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/seed_$seed.out 2>&1 &
done
wait