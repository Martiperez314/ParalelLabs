#!/bin/bash
#SBATCH --job-name=partis_async
#SBATCH --output=partis_async.out
#SBATCH --error=partis_async.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00

source /home/u258340/opt/nvidia/hpc_sdk/modulefiles/nvhpc/255.sh
make
./partis_oacc_async 1000 1