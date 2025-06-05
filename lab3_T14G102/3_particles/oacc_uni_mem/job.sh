#!/bin/bash
#SBATCH --job-name=partikel_sim
#SBATCH --output=job_%j.out
#SBATCH --error=job_error.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load conda
conda create -n name_default_conda
conda activate name_default_conda
conda install matplotlib opencv
python plot.py

# Modul für GPU-Compiler laden (anpassen je nach Cluster)
module load gcc
module load nvidia/acc

# Profiling mit Nsight Systems
nsys profile -o particle_profile_report ./partis_seq 1000 0
