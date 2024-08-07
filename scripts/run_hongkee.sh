#!/bin/bash
#SBATCH -J hongkee_test_md
#SBATCH -p cas_v100nv_8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time 47:30:00
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

srun python $WORKSPACE_PATH/main_md_hongkee.py --gpu-list 0

echo "DONE"
