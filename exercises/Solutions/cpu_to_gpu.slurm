#!/bin/bash
#SBATCH --job-name=cpu_to-gpu
#SBATCH --nodes=1                
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
##SBATCH --reservation=training

ml WebProxy

source activate_venv hpc_ai_env

python cpu_to_gpu.py
