#!/bin/bash
#SBATCH --job-name=vecadd_seq
#SBATCH --output=vecadd_seq%j.out
#SBATCH --error=vecadd_seq%j.err
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --partition=std

make >> make.out || exit 1 

./vecadd_seq 5000000