#!/bin/bash
# This runs transformers only. More changes are needed in scripts to allow XGB,RandomForest, and baselines
for task in risk_profiling sentiment_analysis; do
    for dir in $(ls ./configs/$task);do
        echo "Running $dir configuration for $task:"
        python train.py -config_dir configs/$task/$dir;
        rm -r results/$task/$dir/ckpts
    done
done
