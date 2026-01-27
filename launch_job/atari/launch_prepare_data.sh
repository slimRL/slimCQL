GAME="CrazyClimber"
FIRST_SEED=1
LAST_SEED=5

sbatch --job-name prepare_data-$GAME --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=2 --mem-per-cpu=15G --time=4:00:00 --partition amd,amd2,amd3 \
--output=experiments/atari/logs/prepare_data_$GAME/seeds_%a.out \
launch_job/atari/prepare_data.sh $GAME
