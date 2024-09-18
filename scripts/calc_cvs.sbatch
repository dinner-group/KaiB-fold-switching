#!/bin/bash
 
#SBATCH --job-name=cvs
#SBATCH --output=cvs.out
#SBATCH --error=cvs.err
#SBATCH --partition=dinner
#SBATCH --account=pi-dinner
#SBATCH --qos=dinner
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48

python calc_cvs.py $1 $2 $3 $4
