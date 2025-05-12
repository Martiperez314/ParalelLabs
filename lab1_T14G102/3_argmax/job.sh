#!/bin/bash

# SLURM job configuration for running argmax.c
#SBATCH --job-name=argmax
#SBATCH -p std
#SBATCH --output=out_argmax_%j.out
#SBATCH --error=out_argmax_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:01:00

make >> make.out || exit 1     

# Executar amb 8 threads i K = 2048
./argmax 8 2048
