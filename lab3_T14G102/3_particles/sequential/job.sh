#!/bin/bash
#SBATCH --job-name=partis_seq
#SBATCH --output=partis_seq.out
#SBATCH --error=partis_seq.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00

make clean
make all
./partis_seq 50000 0