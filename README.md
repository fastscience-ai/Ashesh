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


## 250222 Update  

### Brief summary :
**Environment setup**
* This repo follows the <code> stable-material-diffsion </code> [repo](#https://github.com/Lactobacillus/stable-material-diffusion). Please follow the instructions below : 
    ```
    conda create -n smd python=3.11
    conda activate smd
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

    pip install wandb==0.19.1 matplotlib==3.10.0 h5py==3.12.1 jupyter==1.1.1 ase==3.24.0 pymatgen==2024.11.13 ujson==5.10.0 msgpack==1.1.0 schedulefree==1.4 easydict==1.13 rdkit==2024.3.2 permissive-dict==1.0.4

    pip install tqdm argparse scipy 
    
    # Optional : If you want to sync with the original repo, 
    pip install torch_geometric==2.6.1 pyg_lib==0.4.0 torch_scatter==2.1.2 torch_sparse==0.6.18 torch_cluster==1.6.3 torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-2.5.1%2Bcu124.html
    
    ```
    
* Currently all the experiments are running based on <code> main_md_lazy_train.py </code> and  <code> main_md_multi.py </code>.  To run training, run <code>./scripts/linformer_lazy.sh</code>. 
* Additionaly refer to functions <code>model_pbc_call, one_step_loss_unified, pbc_coord</code> in  <code>utils/functions.py</code>, <code>sample_one_step_unified</code> in <code>sampler.py</code>. 

<br>

## 250328 Update 

### Train and Inference
**to train :** run ```/scripts/linformer_lazy_train.sh```. <br>
**to generate samples :** run ```/scripts/linformer_lazy_eval.sh```. <br>
Currently evaluation codes are part of the training code, which will divide into separate functions which targets for multi-frame inference.



**Options**
```
--how_to_sample : sampling method. 
                one_step : predicts X(K, t) - X(K, 0)
                one_step_diff : predicts X(K+1, 0) - X(K, 0)
--temperature : temperatures to train. give arguments as T_1 T_2 T_3...
--t_selection : currently deprecated, will be used in sampling process.
--t_to_simulate : temprature to simulate. 
```
