#!/bin/bash
#SBATCH --job-name=fc_mpi_test
#SBATCH --output=fc_mpi_test.out
#SBATCH --error=fc_mpi_test.err
#SBATCH --ntasks=4
#SBATCH --time=00:05:00
#SBATCH --partition=std

module load gcc/13.3.0
module load openmpi/4.1.1
mpirun \
  --mca btl ^openib \
  ./fc_mpi input_planes/input_planes_test.txt 5 0 1