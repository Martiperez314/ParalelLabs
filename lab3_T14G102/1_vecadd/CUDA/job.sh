#!/bin/bash
#SBATCH --job-name=vecadd_cuda
#SBATCH --output=cuda%j.out
#SBATCH --error=cuda%j.err
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --partition=std
#SBATCH -p gpu
#SBATCH --gres=gpu:1

module load nvhpc/24.9

make >> make.out || exit 1 

./vecadd_cuda 10000