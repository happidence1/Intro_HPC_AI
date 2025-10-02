#!/bin/bash
#SBATCH --job-name=FIXME       # Change the job name to be gpu related 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=FIXME      # Change the partition to be gpu
#SBATCH --gres=gpu:h100:FIXME   # Request 1 H100 GPU
#SBATCH --cpus-per-task=FIXME  # Set number of CPU cores per task (e.g., 8)
#SBATCH --mem=32G
#SBATCH --time=01:00:00
##SBATCH --reservation=training

module load WebProxy

source activate_venv hpc_ai_env

# ---- Run Training ----
# use modified GPU training script
python FIXME
