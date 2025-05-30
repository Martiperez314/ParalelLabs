#!/bin/bash
#SBATCH --job-name=partis_seq
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00

make
./partis_seq 1000 1