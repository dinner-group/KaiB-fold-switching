#!/bin/bash
 
#SBATCH --job-name=omegas
#SBATCH --output=omegas.out
#SBATCH --error=omegas.err
#SBATCH --partition=dinner
#SBATCH --account=pi-dinner
#SBATCH --qos=dinner
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G

folder=$1
header=$2
END=$3
output_folder="${folder}/outputs"
echo "Input folder: $folder"
echo "Output folder: $outputs"

if [ ! -d $output_folder ]; then
    mkdir $output_folder
fi

# Load the default version of GNU parallel.
module load parallel
 
# When running a large number of tasks simultaneously, it may be
# necessary to increase the user process limit.
ulimit -u 10000
 
# This specifies the options used to run srun. The "-N1 -n1" options are
# used to allocates a single core to each task.
srun="srun --exclusive -N1 -n1"
 
parallel="parallel --delay 0.2 -j $SLURM_NTASKS --joblog runtask.log"
 
echo "  started at `date`"
script="/project/dinner/scguo/kaiB/scripts/calc_omegas.py"
 
# run the script for every file q_X_Y.txt where X runs from 2-8 and Y runs from 0-9 (inclusive)
$parallel "$srun python $script $folder/$header.run.00.up $folder/$header.run.{1}.up $output_folder/$header.run.{1}" ::: $(seq -f "%02g" 0 $END)
