#!/bin/bash
#SBATCH --job-name=matmul
#SBATCH --output=out_matmul%j.out
#SBATCH --error=out_matmul%j.err
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1

module load nvhpc/24.9

make >> make.out || exit 1

for N in 128 256 512 1024 2048; do
    echo "***Running MM with size ${N}x${N}***"
    ./matmul $N 1
    echo ""
done