#!/bin/bash

#SBATCH --job-name=kaiB_cis

#SBATCH --partition=dinner
#SBATCH --account=pi-dinner
#SBATCH --qos=dinner

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --array=0-

#SBATCH --time=1:00:00


jobindex=$SLURM_ARRAY_TASK_ID

micromamba activate md
source /project/dinner/scguo/upside2/sourceme.sh

home_dir="/project/dinner/scguo/kaiB"
run_script=$home_dir/dga/run_dga.py

# use the array id to get the correct name of the seed file
seedfile=$home_dir/dga/seeds_remd/files.txt
coordfile="$(sed "${jobindex}q;d" $seedfile)"
coordhead="${coordfile%.*}"

iso_params_file=$1
temperature=$2

base_dir=$home_dir/dga/new_$temperature
if [ ! -d $base_dir ]; then
    mkdir $base_dir
fi

echo "Isomerization params file $iso_params_file"
echo "Temperature = 0.$temperature"

iso="cis"
run_dir=$base_dir/${coordhead}_${iso}
if [ ! -d $run_dir ]; then
    mkdir $run_dir
fi
echo "Launching run $run_dir"
python $run_script -p 4 -i $iso_params_file -t 0.$temperature -c $coordfile -s dga -o $run_dir -m 2000000
