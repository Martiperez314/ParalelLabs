#!/bin/bash

# Configuration for 1 node, 4 cores and 5 minutes of execution time
#SBATCH --job-name=ex1
#SBATCH -p std
#SBATCH --output=out_montecarlo_%j.out
#SBATCH --error=out_montecarlo_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:05:00

make >> make.out || exit 1      # Exit if make fails

./montecarlo 