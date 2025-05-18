#!/bin/bash
#SBATCH --job-name=fc_mpi
#SBATCH --output=fc_mpi_test%j.out
#SBATCH --error=fc_mpi_test%j.err
#SBATCH --ntasks=4
#SBATCH --time=00:05:00
#SBATCH --partition=std

module load gcc/13.3.0
module load openmpi/4.1.1

mpirun -n 4 ./fc_mpi input_planes_test.txt 25 0 1