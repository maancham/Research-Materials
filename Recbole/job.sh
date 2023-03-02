#!/bin/bash
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=0
#SBATCH --time=1-03:00
#SBATCH --account=rrg-smucker
module load python cuda/11.4  # CUDA must be loaded if using ZeRO offloading to CPU or NVMe. Version must be the same used to compile PyTorch.
source /home/maanch/projects/rrg-smucker/maanch/Recbole/env/bin/activate


export NCCL_BLOCKING_WAIT=1  # Pytorch Lightning uses the NCCL backend for inter-GPU communication by default. Set this variable to avoid timeout errors.
export TRANSFORMERS_OFFLINE=1
# PyTorch Lightning will query the environment to figure out if it is running inside a SLURM batch job
# If it is, it expects the user to have requested one task per GPU.
# If you do not ask for 1 task per GPU, and you do not run your script with "srun", your job will fail!
python trainer.py ml-25m