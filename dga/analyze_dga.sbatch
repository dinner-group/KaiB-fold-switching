#!/bin/bash
 
#SBATCH --job-name=analyze
#SBATCH --output=analyze.out
#SBATCH --error=analyze.err
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
 
cd $1
echo "  started at `date`"
script="/project/dinner/scguo/kaiB/scripts/analyze_upside_trajs.sh"
 
# run the script for every file q_X_Y.txt where X runs from 2-8 and Y runs from 0-9 (inclusive)
analyze() {
    if [ -d $1/${1}${2}/$3 ]; then
        cd $1/${1}${2}/${3}
        bash /project/dinner/scguo/kaiB/scripts/analyze_upside_trajs.sh .
        cd ../../../
    fi
}
export -f analyze
$parallel analyze ::: {00..12} ::: {000..999} ::: cis trans

echo "  finished at `date`"

cd ../
