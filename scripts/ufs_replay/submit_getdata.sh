#!/bin/bash

#SBATCH -J 2005
#SBATCH -o slurm/getdata.%j.out
#SBATCH -e slurm/getdata.%j.err
#SBATCH --nodes=1
#SBATCH --nodelist=linlincui-awscpueast1f-00002-1-0010
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=compute
#SBATCH -t 120:00:00


source /contrib/Linlin.Cui/miniforge3/etc/profile.d/conda.sh
conda activate mlsfs
python get_data_to_zarr_slurm.py 2005
