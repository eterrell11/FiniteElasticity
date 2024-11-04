#!/bin/bash

#SBATCH --job-name=twist_FE
#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --output=twist_out
#SBATCH --error=twist_error
#SBATCH --mem=5GB
#SBATCH --time=3-00:00:00
#SBATCH --mail-user=terrell1@email.unc.edu 

sh cook_parallel_refine
#matlab -nodesktop -nosplash -nosoftwareopengl -singleCompThread -r error_analysis -logfile error_analysis_log.out
