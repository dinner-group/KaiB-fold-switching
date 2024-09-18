#!/bin/bash

echo "Started at `date`"

# zsh
# conda activate py39
# source /project/dinner/scguo/upside2/sourceme.sh

# input folder
folder=$1
output_folder="${folder}/outputs"
echo "Input folder: $folder"
echo "Output folder: $outputs"

if [ ! -d $output_folder ]; then
    mkdir $output_folder
fi

for f in $folder/*.up; do
    header="${f%.*}"
    datafile="${output_folder}/$header"
    echo "Analyzing $f..."
    python $UPSIDE_HOME/py/get_info_from_upside_traj.py $f $datafile
done

echo "Done at `date`"
