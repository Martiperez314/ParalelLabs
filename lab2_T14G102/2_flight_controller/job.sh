#!/bin/bash
#SBATCH --job-name=fc_mpi_test        
#SBATCH --output=fc_mpi_test.out       
#SBATCH --error=fc_mpi_test.err        
#SBATCH --ntasks=4                     
#SBATCH --time=00:05:00                

mpirun ./fc_mpi input_planes/input_planes_test.txt 5 0 1
