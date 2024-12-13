# Ashesh: Diffusion+MD
This is the pilot code for experimenting stable diffusion model for generating MD trajectory
## install conda <br />
Go to https://docs.conda.io/en/latest/miniconda.html <br />
Then download proper installer for your OS  <br />
`bash [proper installer] ` <br />
## Install 
conda env create --file=environments.yml
## MD Diffusion
`python main_md.py `
## Dataset
https://drive.google.com/drive/folders/1FawjldF4ZslwzdzaNXaa42s5wG3vrgwd?usp=sharing
## MD Dataset
https://drive.google.com/drive/folders/1SfYbW-jTnvWkR93am-c3PyumSZo1AQ9E?usp=sharing
## MD Long_run result
https://drive.google.com/drive/folders/1SfYbW-jTnvWkR93am-c3PyumSZo1AQ9E?usp=sharing
# How to setup 
conda create -n md_diffusion python=3.11
conda activate md_diffusion

conda install -n md_diffusion pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

conda install -n md_diffusion pyg=2.4.0 -c pyg

conda install -n md_diffusion -c huggingface transformers=4.32.1

conda install -n md_diffusion -c anaconda matplotlib=3.8.0 h5py=3.9.0 jupyterlab=3.6.3

conda install -n md_diffusion -c conda-forge wandb=0.16.1 ase=3.22.1 gpaw=23.9.1 openbabel=3.1.1 pymatgen=2024.2.23 ujson=5.9.0

pip install git+https://github.com/bluehope/egnn-pytorch.git@PBC_support

pip install git+https://github.com/bluehope/equiformer-pytorch.git@PBC_support

conda install -n md_diffusion -c conda-forge netCDF4
