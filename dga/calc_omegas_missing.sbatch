#!/bin/bash
 
#SBATCH --job-name=analyze_omegas
#SBATCH --output=omegas.out
#SBATCH --error=omegas.err
#SBATCH --partition=dinner
#SBATCH --account=pi-dinner
#SBATCH --qos=dinner

#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --mem-per-cpu=4G

source ~/.bashrc
source ~/.zshrc
zsh
micromamba activate md
source /project/dinner/scguo/upside2/sourceme.sh

# Load the default version of GNU parallel.
module load parallel
 
# When running a large number of tasks simultaneously, it may be
# necessary to increase the user process limit.
ulimit -u 10000
 
# This specifies the options used to run srun. The "-N1 -n1" options are
# used to allocates a single core to each task.
srun="srun --exclusive -N1 -n1"
 
# This specifies the options used to run GNU parallel:
#
#   --delay of 0.2 prevents overloading the controlling node.
#
#   -j is the number of tasks run simultaneously.
#
#   The combination of --joblog and --resume create a task log that
#   can be used to monitor progress.
parallel="parallel --delay 0.2 -j $SLURM_NTASKS --joblog runtask.log"
 
echo "  started at `date`"
script=/project/dinner/scguo/kaiB/scripts/calc_omegas.py

analyze() {
    if [ -f ${1}/${1}${2}/outputs/${1}${2}_${3}_fs_qtots.pkl ]; then
        if [ ! -f ${1}/${1}${2}/outputs/${1}${2}_${3}_Omega.npy ]; then
            echo "Computing ${1}/${1}${2}/outputs/${1}${2}_${3}_Omega.npy" 
            python /project/dinner/scguo/kaiB/scripts/calc_omegas.py ${1}/${1}${2}/${3}/*.up ${1}/${1}${2}/${3}/*.up ${1}/${1}${2}/outputs/${1}${2}_${3}
        fi
    fi
}
export -f analyze

cd $1
# run the script for every DGA output directory
$parallel analyze ::: {00..12} ::: {000..999} ::: cis trans
cd ../ 
echo "  finished at `date`"
