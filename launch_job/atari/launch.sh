launch_job/atari/local_cql.sh --experiment_name test_VideoPinball --first_seed 1 --last_seed 1 \
    --data_dir "experiments/atari/datasets/slim_dataset" --architecture_type cnn --features 32 64 64 512 \
    --n_buffers_to_load 3 --replay_buffer_capacity 100_000 --n_epochs 50 --n_fitting_steps 62_500 \
    --target_update_period 2_000 --batch_size 32 --update_horizon 1 --gamma 0.99 --learning_rate 5e-5 \
    --alpha_cql 0.1 --disable_wandb