#!/bin/bash
 
#SBATCH --job-name=contacts
#SBATCH --output=contacts.out
#SBATCH --error=contacts.err
#SBATCH --partition=dinner
#SBATCH --account=pi-dinner
#SBATCH --qos=dinner
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

python calc_contacts.py $1 $2 $3 $4
