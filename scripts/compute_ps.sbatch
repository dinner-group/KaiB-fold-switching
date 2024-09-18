#!/bin/bash
 
#SBATCH --job-name=ps
#SBATCH --output=ps.out
#SBATCH --error=ps.err
#SBATCH --partition=dinner
#SBATCH --account=pi-dinner
#SBATCH --qos=dinner
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

./compute_ps.sh $1 $2
