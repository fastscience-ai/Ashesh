#!/bin/bash
#SBATCH -J MDMD_multi
#SBATCH -p eme_h200nv_8
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

srun python $WORKSPACE_PATH/main_md_lazy_train.py\
 --how_to_sample "one_step_diff" \
 --learning_rate 5e-4\
 --n_offset 10 \
 --num_epochs 7001\
 --save_interval 250\
 --temperature 200 300 500 800 \
 --t_selection 200 300 500 800 \
 --t_to_simulate 1000


echo "DONE"
