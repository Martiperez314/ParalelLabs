#!/bin/bash
#SBATCH --job-name=partis_managed
#SBATCH --output=partis_managed.out
#SBATCH --error=partis_managed.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00

export NVHPC_HOME=$HOME/nvhpc/Linux_x86_64/25.5
export PATH=$NVHPC_HOME/compilers/bin:$PATH
export LD_LIBRARY_PATH=$NVHPC_HOME/compilers/lib:$LD_LIBRARY_PATH

make
./partis_oacc_managed 1000 0
