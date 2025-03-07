#!/bin/bash

#SBATCH --job-name=test
#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=test_out
#SBATCH --error=test_error
#SBATCH --mem=1GB
#SBATCH --time=3-00:00:00
#SBATCH --mail-user=terrell1@email.unc.edu

srun translate_bdf2
matlab -nodesktop -nosplash -singleCompThread -r error_analysis
