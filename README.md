# Machine Learning for Seasonal Forecast System (SFS)

## Overview
This project utilizes Spherical Fourier Neural Operators (SFNO)](https://arxiv.org/pdf/2306.03838) developed by NVIDIA. 

## Installation
The recommended way to setup the environment for installing `mlsfs` is to use `conda` with the environment.yml:
```bash
conda env create -f environment.yml
conda activate mlsfs
````

And pytorch-cuda libray must be installed. The install command can be obtained from [PyTorch](https://pytorch.org/get-started/locally/) 
based on cuda version of your system. For example, for cuda-12.4, the install command is: 
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

If you want to actively develop mlsfs, we recommend building it in your environment from github:
```bash
git clone git@github.com:NOAA-EMC/mlsfs.git
cd mlsfs
pip install -e .
````
