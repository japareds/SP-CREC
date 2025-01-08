#!/bin/bash

#SBATCH --chdir=/scratch/nas/3/jparedes/IRNet/src
#SBATCH --job-name="IRNet_design"
#SBATCH --output=/scratch/nas/3/jparedes/IRNet/outputs/output-IRNet_design-%j-%N.out
#SBATCH --error=/scratch/nas/3/jparedes/IRNet/errors/error-IRNet_design-%j-%N.out

epsilon=5e-2
num_it=20

python network_design_SST.py --design_large_networks -e ${epsilon} -n_it ${num_it}