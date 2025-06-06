#!/bin/bash
#SBATCH --job-name=partikel_sim
#SBATCH --output=job_%j.out
#SBATCH --error=job_error.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module purge
module load nvhpc/24.9

make >> make.out || exit 1

nsys profile -o particle_profile_report ./partis_oacc_uni_mem 1000 0