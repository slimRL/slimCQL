#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

source env/bin/activate
export XLA_PYTHON_CLIENT_MEM_FRACTION=$(echo "scale=2 ; 0.98 / ($LAST_SEED - $FIRST_SEED + 1)" | bc)


for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    python3 experiments/$ENV_NAME/$ALGO_NAME.py --experiment_name $EXPERIMENT_NAME --seed $seed $ARGS >> experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/seed_$seed.out 2>&1 &
done
wait

N_PARALLEL_EPOCHS=3
N_EPOCHS=$(echo "$ARGS" | grep -oP '(?<=--n_epochs |-ne )\d+')

FRACTION_GPU=$(echo "scale=2 ; 0.98 / $N_PARALLEL_EPOCHS" | bc)
export XLA_PYTHON_CLIENT_MEM_FRACTION=$FRACTION_GPU

for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    epoch=0
    while (( epoch < N_EPOCHS+1 ))
    do
        python3 experiments/$ENV_NAME/evaluate.py --experiment_name $EXPERIMENT_NAME --algo_name $ALGO_NAME --seed $seed --epoch $epoch >> experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/seed_$seed.out 2>&1 &
        
        ((epoch++))
        
        if (( epoch % N_PARALLEL_EPOCHS == 0 )); then
            wait
        fi
    done
    wait
    python3 experiments/$ENV_NAME/synchronize_evaluation_wandb.py --experiment_name $EXPERIMENT_NAME --algo_name $ALGO_NAME --seed $seed --env_name "atari" --delete_models >> experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/seed_$seed.out 2>&1 &
done
wait