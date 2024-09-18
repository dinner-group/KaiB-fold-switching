#!/bin/bash

N=$1 # should be 0-7, corresponding to 8 groups of 1536 starting points
temperature=$2

home_dir="/project/dinner/scguo/kaiB"
base_dir=$home_dir/dga/new_$temperature
if [ ! -d $base_dir ]; then
    mkdir $base_dir
fi

if [ ! -d $home_dir/dga/logs/$temperature/$N ]; then
    mkdir $home_dir/dga/logs/$temperature/$N
fi


sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=kaiB_${temperature}_$N
#SBATCH --output=$home_dir/dga/logs/${temperature}/$N/%a.out
#SBATCH --error=$home_dir/dga/logs/${temperature}/$N/%a.err

#SBATCH --partition=dinner
#SBATCH --account=pi-dinner
#SBATCH --qos=dinner

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=$3

#SBATCH --time=48:00:00

jobindex=\$SLURM_ARRAY_TASK_ID
echo \$jobindex

source ~/.bashrc
source ~/.zshrc
zshrc
micromamba activate md
source /project/dinner/scguo/upside2/sourceme.sh
home_dir="/project/dinner/scguo/kaiB"
run_script=\$home_dir/dga/run_dga_single.py
seedfile=\$home_dir/dga/seeds_remd/files.txt

# each jobarray is (12888 / 8) = 1536 starting points
# each job in jobarray runs 4 starting points
start_coord_index=\$((1536 * $N + jobindex * 4))

iso_params_file=/project2/dinner/darrenjl/kaiB/iso_params/barrier/barrier_5.0_cis.dat

echo "Isomerization params file \$iso_params_file"
echo "Temperature = 0.$temperature"
base_dir=\$home_dir/dga/new_$temperature

for i in {0..3}; do
    for iso in cis trans ; do
        if [ \$iso = "cis" ]; then
            p=3
        elif [ \$iso = "trans" ]; then
            p=4
        fi
        # use the array id to get the correct name of the seed file
        coordfile_index=\$((start_coord_index + i + 1))
        coordfile="\$(sed "\${coordfile_index}q;d" \$seedfile)"
        echo "\$coordfile_index \$coordfile \$iso"
        coordhead="\${coordfile%.*}"
        
        index="\$(printf "%05d\n" \$coordfile_index)"
        run_dir=\$base_dir/\${index}
        if [ ! -d \$run_dir ]; then
            mkdir \$run_dir
        fi
        echo "Launching run \$run_dir"
        srun --export=ALL python \$run_script -p \$p -i \$iso_params_file -t 0.$temperature -c \$coordhead -s \$iso -o \$run_dir -m 200000
    done
done
EOT
