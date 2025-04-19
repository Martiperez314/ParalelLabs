#!/bin/bash

# Configuration for 1 node, 4 cores and 30 seconds of execution time
#SBATCH --job-name=ex2
#SBATCH -p std
#SBATCH --output=out_histogram_%j.out
#SBATCH --error=out_histogram_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:00:30

make >> make.out || exit 1      # Exit if make fails

./histogram

