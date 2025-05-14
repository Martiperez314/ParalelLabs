#!/bin/bash
#SBATCH --job-name=job
#SBATCH --output=out_scaling_%j.out
#SBATCH --error=err_scaling_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=192
#SBATCH --nodes=4
#SBATCH --cpus-per-task=1

module purge
module load gcc/13.3.0 openmpi/5.0.3

make >> make.out || exit 1

dimension=10
seed=40

echo "Strong Scaling (100000000 samples total):"
for p in 1 2 4 8 16 32 64 96 128 192; do
    echo "Num of processes: $p"
    srun --ntasks=$p --ntasks-per-node=192 ./montecarlo $dimension 100000000 $seed
done

echo "---------------------------------------------"
echo "Weak Scaling (100000000 samples per process):"
for p in 1 2 4 8 16 32 64 96 128 192; do
    samples=$((100000000 * p))
    echo "Processes: $p  |  Samples: $samples"
    srun --ntasks=$p --ntasks-per-node=192 ./montecarlo $dimension $samples $seed