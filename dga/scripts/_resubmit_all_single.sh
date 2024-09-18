#!/bin/bash

N=$1
temperature=$2
iso=$3
if [ $iso = "cis" ]; then
    p=3
elif [ $iso = "trans" ]; then
    p=4
fi

home_dir="/project/dinner/scguo/kaiB"
base_dir=$home_dir/dga/new_$temperature
if [ ! -d $base_dir ]; then
    mkdir $base_dir
fi

if [ ! -d $home_dir/dga/logs/$iso/$temperature ]; then
    mkdir $home_dir/dga/logs/$iso/$temperature
fi


sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=kaiB_${iso}_${temperature}
#SBATCH --output=$home_dir/dga/logs/${iso}/${temperature}/%a.out
#SBATCH --error=$home_dir/dga/logs/${iso}/${temperature}/%a.err

#SBATCH --partition=dinner
#SBATCH --account=pi-dinner
#SBATCH --qos=dinner

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=$N

#SBATCH --time=6:00:00

jobindex=\$SLURM_ARRAY_TASK_ID
echo \$jobindex

source ~/.bashrc
source ~/.zshrc
zshrc
micromamba activate md
source /project/dinner/scguo/upside2/sourceme.sh

home_dir="/project/dinner/scguo/kaiB"
run_script=\$home_dir/dga/run_dga.py

# use the array id to get the correct name of the seed file
seedfile=\$home_dir/dga/seeds_remd/files.txt
coordfile="\$(sed "\${jobindex}q;d" \$seedfile)"
echo \$coordfile
coordhead="\${coordfile%.*}"

iso_params_file=/project2/dinner/darrenjl/kaiB/iso_params/barrier/barrier_5.0_cis.dat

echo "Isomerization params file \$iso_params_file"
echo "Temperature = 0.$temperature"

base_dir=\$home_dir/dga/new_$temperature
index="\$(printf "%05d\n" \$jobindex)"
run_dir=\$base_dir/\${index}
if [ ! -d \$run_dir ]; then
    mkdir \$run_dir
fi
echo "Launching run \$run_dir"
srun --export=ALL python \$run_script -p $p -i \$iso_params_file -t 0.$temperature -c \$coordhead -s $iso -o \$run_dir -m 200000
EOT
