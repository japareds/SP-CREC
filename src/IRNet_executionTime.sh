#!/bin/bash

#SBATCH --chdir=/scratch/nas/3/jparedes/IRNet/src
#SBATCH --job-name="IRNet_executionTime"
#SBATCH --output=/scratch/nas/3/jparedes/IRNet/outputs/output-IRNet_executionTime-%j-%N.out
#SBATCH --error=/scratch/nas/3/jparedes/IRNet/errors/error-IRNet_executionTime-%j-%N.out

python irnet_executionTime_random.py