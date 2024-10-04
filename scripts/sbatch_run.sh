#!/bin/bash

#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=fcn
#SBATCH --output=output.txt
#SBATCH --error=error.txt

module use /nvidia/nvhpc/24.7/modulefiles
module load nvhpc-hpcx-cuda12/24.7

#activate conda environment

source /lustre/Linlin.Cui/miniforge3/etc/profile.d/conda.sh
conda activate myenv

MPI_HOSTS=$(scontrol show hostname $SLURM_NODELIST | sed "s/$/:$SLURM_NTASKS_PER_NODE/" | xargs | sed 's/ /,/g')
nodes=$( scontrol show hostname $SLURM_NODELIST )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

mpirun -np ${SLURM_NPROCS} \
-host ${MPI_HOSTS} \
-x MASTER_ADDR=${head_node_ip} \
-x MASTER_PORT=1234 \
-x PATH \
-bind-to none \
-map-by slot -mca pml ob1 \
-mca btl ^openib \
python3 train.py --run_num=1 --yaml_config=./config/SFNO_era5.yaml --config=sfno_backbone
