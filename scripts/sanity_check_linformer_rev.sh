#!/bin/bash
#SBATCH -J mdmd
#SBATCH -p cas_v100nv_4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time 23:30:00
#SBATCH --comment pytorch
#SBATCH -o /scratch/x2895a03/research/md-diffusion/Ashesh/logs/output_%x_%j.out
#SBATCH -e /scratch/x2895a03/research/md-diffusion/Ashesh/logs/output_%x_%j.err

source /home01/$USER/.bashrc
export CONDA_ENVS_PATH=/scratch/x2895a03/.conda/envs
export CONDA_PKGS_DIRS=/scratch/x2895a03/.conda/pkgs
conda activate smd

WORKSPACE_PATH=/scratch/x2895a03/research/md-diffusion/Ashesh

cd $WORKSPACE_PATH

echo "START"

srun python $WORKSPACE_PATH/main_md_vdm.py --gpu-list 0

echo "DONE"
