#!/bin/bash

#SBATCH --job-name=test
#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=test_out
#SBATCH --error=test_error
#SBATCH --mem=5GB
#SBATCH --time=3-00:00:00
#SBATCH --mail-user=terrell1@email.unc.edu
module add gcc
module add matlab
sh main_script_slurm.sh
matlab -nodesktop -nosplash -nosoftwareopengl -singleCompThread -r error_analysis -logfile error_analysis_log.out
