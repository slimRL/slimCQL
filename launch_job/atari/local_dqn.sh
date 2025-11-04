#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

if ! tmux has-session -t slimcql; then
    tmux new-session -d -s slimcql
    echo "Created new tmux session - slimcql"
fi

tmux send-keys -t slimcql "cd $(pwd)" ENTER
tmux send-keys -t slimcql "source env/bin/activate" ENTER
FRACTION_GPU=$(echo "scale=2 ; 0.98 / ($LAST_SEED - $FIRST_SEED + 1)" | bc)
tmux send-keys -t slimcql "export XLA_PYTHON_CLIENT_MEM_FRACTION=$FRACTION_GPU" ENTER

echo "launch train $ALGO_NAME local"
for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    tmux send-keys -t slimcql \
    "python3 experiments/$ENV_NAME/$ALGO_NAME.py --experiment_name $EXPERIMENT_NAME --seed $seed $ARGS >> experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/seed_$seed.out 2>&1 &" ENTER
done
tmux send-keys -t slimcql "wait" ENTER

N_PARALLEL_EPOCHS=10
N_EPOCHS=$(echo "$ARGS" | grep -oP '(?<=--n_epochs |-ne )\d+')

FRACTION_GPU=$(echo "scale=2 ; 0.98 / $N_PARALLEL_EPOCHS" | bc)
tmux send-keys -t slimcql "export XLA_PYTHON_CLIENT_MEM_FRACTION=$FRACTION_GPU" ENTER

for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    epoch=0
    while (( epoch < N_EPOCHS+1 ))
    do
        tmux send-keys -t slimcql \
        "python3 experiments/$ENV_NAME/evaluate.py --experiment_name $EXPERIMENT_NAME --algo_name $ALGO_NAME --seed $seed --epoch $epoch >> experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/seed_$seed.out 2>&1 &" ENTER
        
        ((epoch++))
        
        if (( epoch % N_PARALLEL_EPOCHS == 0 )); then
            tmux send-keys -t slimcql "wait" ENTER
        fi
    done
    tmux send-keys -t slimcql "wait" ENTER
    tmux send-keys -t slimcql \
        "python3 experiments/synchronize_evaluation_wandb.py --experiment_name $EXPERIMENT_NAME --algo_name $ALGO_NAME --seed $seed --env_name "atari" --delete_models >> experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/seed_$seed.out 2>&1 &" ENTER
done
tmux send-keys -t slimcql "wait" ENTER