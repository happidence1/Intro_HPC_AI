#!/bin/bash
#SBATCH --job-name=ddp-multi-node
#SBATCH --nodes=2                     # Set Number of nodes greater than 1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2              # Set number of GPUs per node (<= 2)
#SBATCH --ntasks-per-node=2           # Set it to be equal to the number of GPUs per node
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
##SBATCH --reservation=training

ml WebProxy
source activate_venv hpc_ai_env

# Use first hostname from node list for MASTER_ADDR
# Sometimes $SLURM_NODELIST doesn’t resolve to a real address, so use scontrol show hostnames here to be safe
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)

# Dynamically choose a random port to avoid reuse conflicts
export MASTER_PORT=$((12000 + RANDOM % 10000))

# Export NCCL debug flags for multi-node troubleshooting (optional).
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker

# Let PyTorch infer RANK, LOCAL_RANK, WORLD_SIZE from SLURM env vars
# Use srun to handle launching multiple processes across nodes
srun --export=ALL \
     python ddp_multi_node.py
