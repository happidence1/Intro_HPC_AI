#!/bin/bash
#SBATCH --job-name=mnist_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=cpu
#SBATCH --mem=32G
#SBATCH --time=01:00:00
##SBATCH --reservation=training

ml WebProxy
source activate_venv hpc_ai_env

python mnist_cpu.py
