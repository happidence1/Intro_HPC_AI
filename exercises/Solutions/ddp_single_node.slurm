#!/bin/bash
#SBATCH --job-name=ddp-single-node
#SBATCH --nodes=1		        # Set single node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2        # Set number of GPUs (This is per node)
#SBATCH --ntasks-per-node=2     # Recommend setting it equal to the number of GPUs
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
##SBATCH --reservation=training

ml WebProxy

source activate_venv hpc_ai_env

python ddp_single_node.py
