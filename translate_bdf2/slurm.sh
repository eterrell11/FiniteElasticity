#!/bin/bash

#SBATCH --job-name=49IV10
#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=49IV10_out
#SBATCH --error=49IV10_error
#SBATCH --mem=1GB
#SBATCH --time=3-00:00:00
#SBATCH --mail-user=terrell1@email.unc.edu

srun ./translate_bdf2
