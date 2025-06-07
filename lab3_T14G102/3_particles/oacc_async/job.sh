#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=async
#SBATCH --output=out_async_%j.out
#SBATCH --error=out_async_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:00:30

module load nvhpc/24.9

make >> make.out || exit 1  

nsys profile -o nsys_report ./partis_oacc_async 1000 0
