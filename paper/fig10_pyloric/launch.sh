#!/bin/bash
#SBATCH --array=0,1,2,3,4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

python train.py --seed $SLURM_ARRAY_TASK_ID