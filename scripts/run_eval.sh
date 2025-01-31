#! /usr/bin/env bash

# Function to run training for a dataset and room pair
run_training() {
    dataset=$1
    room1=$2
    room2=$3
    python -m src.GsplatLoc_eval --dataset $dataset --rooms $room1 $room2 --num-iters 2000 --disable-viewer &
}

# Replica dataset
run_training Replica room0 room1
run_training Replica room2 office0
run_training Replica office1 office2
run_training Replica office3 office4

# TUM dataset
run_training TUM freiburg1_desk freiburg1_desk2
run_training TUM freiburg1_room freiburg2_xyz
run_training TUM freiburg3_long_office_household

# Wait for all background processes to finish
wait

echo "All training processes have completed."

# to kill pkill -f GsplatLoc_eval.py
