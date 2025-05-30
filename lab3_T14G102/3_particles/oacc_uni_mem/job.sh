#!/bin/bash
#SBATCH --job-name=partikel_sim
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Modul für GPU-Compiler laden (anpassen je nach Cluster)
module load gcc
module load nvidia/acc

# Profiling mit Nsight Systems
nsys profile -o particle_profile_report ./partis_seq 1000 0
