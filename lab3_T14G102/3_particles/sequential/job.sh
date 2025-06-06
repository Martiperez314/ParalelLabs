#!/bin/bash
#SBATCH --job-name=partis_seq
#SBATCH --output=partis_seq.out
#SBATCH --error=partis_seq.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00

module load conda
conda create -n <my_conda_env>
conda activate <my_conda_env>
conda install matplotlib opencv
python plot.py

make >> make.out || exit 1

./partis_seq 50000 0