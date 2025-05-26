#!/bin/bash
#SBATCH --job-name=vecadd_oacc
#SBATCH --output=oacc%j.out
#SBATCH --error=oacc%j.err
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --partition=std


