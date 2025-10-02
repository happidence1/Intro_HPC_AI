#!/bin/bash
#SBATCH --job-name=ddp-single-node
#SBATCH --nodes=FIXME		        # Set single node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:FIXME        # Set number of GPUs (<=2, This is per node)
#SBATCH --ntasks-per-node=FIXME     # Recommend setting it equal to the number of GPUs
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
##SBATCH --reservation=training

ml WebProxy
source activate_venv hpc_ai_env

# ---- Run Training ----
# use modified GPU training script
python FIXME
