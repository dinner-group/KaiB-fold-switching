#!/bin/bash

echo "Started at `date`"

source /project/dinner/scguo/upside2/sourceme.sh

# input folder
folder=$1
pdbname=$2
topfile="${folder}/inputs/$pdbname-HDX.up"
output_folder="${folder}/outputs"
echo "Input folder: $folder"
echo "Output folder: $outputs"

if [ ! -d $output_folder ]; then
    mkdir $output_folder
fi

i=00
for f in $folder/*.up; do
    header="${*/%f%.*}"
    datafile="$output_folder/${i}_ps.npy"
    echo "Analyzing $f..."
    python $UPSIDE_HOME/py/get_protection_state.py $topfile $f $datafile
    i=$((i+1))
done

echo "Done at `date`"
