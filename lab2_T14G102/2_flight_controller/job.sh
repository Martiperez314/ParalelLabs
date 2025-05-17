#!/bin/bash
#SBATCH --job-name=ej2
#SBATCH --output=job%j.txt
#SBATCH --error=job%j.txt
#SBATCH --ntasks=4
#SBATCH --time=00:05:00
#SBATCH --partition=std

# Load necessary modules
module load gcc/13.3.0
module load openmpi/4.1.1

# Run the MPI executable
mpirun ./fc_mpi input_planes_test.txt 5 0 0