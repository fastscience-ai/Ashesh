#!/bin/bash
#SBATCH -J md_gen
#SBATCH -p eme_h200nv_8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time 11:30:00
#SBATCH --comment pytorch
#SBATCH -o /scratch/x2895a03/research/md-diffusion/Ashesh/logs/output_%x_%j.out
#SBATCH -e /scratch/x2895a03/research/md-diffusion/Ashesh/logs/output_%x_%j.err

source /home/$USER/.bashrc
export CONDA_ENVS_PATH=/home/seunghoon/.conda/envs
export CONDA_PKGS_DIRS=/home/seunghoon/.conda/pkgs
conda activate smd

WORKSPACE_PATH=/home/seunghoon/Ashesh/

cd $WORKSPACE_PATH

echo "START"

# 200K
python $WORKSPACE_PATH/md_simul/MD_300K_exmaple.py --temperature 200
python $WORKSPACE_PATH/md_simul/reading_traj_example.py --temperature 200
# 300K
python $WORKSPACE_PATH/md_simul/MD_300K_exmaple.py --temperature 300
python $WORKSPACE_PATH/md_simul/reading_traj_example.py --temperature 300
# 400K
python $WORKSPACE_PATH/md_simul/MD_300K_exmaple.py --temperature 400
python $WORKSPACE_PATH/md_simul/reading_traj_example.py --temperature 400
# 500K
python $WORKSPACE_PATH/md_simul/MD_300K_exmaple.py --temperature 500
python $WORKSPACE_PATH/md_simul/reading_traj_example.py --temperature 500
# 600K
python $WORKSPACE_PATH/md_simul/MD_300K_exmaple.py --temperature 600
python $WORKSPACE_PATH/md_simul/reading_traj_example.py --temperature 600
# 700K
python $WORKSPACE_PATH/md_simul/MD_300K_exmaple.py --temperature 700
python $WORKSPACE_PATH/md_simul/reading_traj_example.py --temperature 700
# 800K
python $WORKSPACE_PATH/md_simul/MD_300K_exmaple.py --temperature 800
python $WORKSPACE_PATH/md_simul/reading_traj_example.py --temperature 800
# 900K
python $WORKSPACE_PATH/md_simul/MD_300K_exmaple.py --temperature 900
python $WORKSPACE_PATH/md_simul/reading_traj_example.py --temperature 900
# 1000K
python $WORKSPACE_PATH/md_simul/MD_300K_exmaple.py --temperature 1000
python $WORKSPACE_PATH/md_simul/reading_traj_example.py --temperature 1000


echo "DONE"
