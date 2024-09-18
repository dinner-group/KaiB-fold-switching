#!/bin/bash

# conda activate py39
# source /project/dinner/scguo/upside2/sourceme.sh

home_dir="/project/dinner/scguo/kaiB"
run_script=$home_dir/run_md3.py
# pdb_dir="." 
iso_params_file=$1
temperature=$2

base_dir=$home_dir/dga/$temperature
if [ ! -d $base_dir ]; then
    mkdir $base_dir
fi


echo "Isomerization params file $iso_params_file"
echo "Temperature = 0.$temperature"

for i in {05..06}; do
    for j in {00..31}; do
        # do cis and trans
        # make output directory
        iso="cis"
        run_dir=$base_dir/${i}_${j}_${iso}
        if [ ! -d $run_dir ]; then
            mkdir $run_dir
        fi
        # launch on Beagle3
        echo "Launching run $run_dir"
        python $run_script -b -n 2 -p 1 -i $iso_params_file -t 0.$temperature -c ${i}_${j} -s dga -o $run_dir -m 2000000

        iso="trans"
        run_dir=$base_dir/${i}_${j}_${iso}
        if [ ! -d $run_dir ]; then
            mkdir $run_dir
        fi
        echo "Launching run $run_dir"
        # launch on dinner
        python $run_script -b -n 2 -p 0 -i $iso_params_file -t 0.$temperature -c ${i}_${j} -s dga -o $run_dir -m 2000000
    done
done

