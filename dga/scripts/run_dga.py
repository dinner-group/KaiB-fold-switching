import numpy as np
import sys, os, shutil
import subprocess as sp
import numpy as np
import tables as tb
from math import sqrt
import time
import argparse

# Run a single long trajectory for DGA
# Should be used with a job array
parser = argparse.ArgumentParser(description="Run a set of DGA Upside simulations using a group of starting files")

parser.add_argument(
    "-i",
    "--iso_param",
    default="",
    help="(str: '') If applicable, paramteter file for the proline isomerization",
)
parser.add_argument(
    "-t",
    "--temp",
    default=None,
    help="(float: None) Temperature for a constant T simulation. If not specified, revert to melting T ramp",
)
parser.add_argument(
    "-o",
    "--output",
    default="./output",
    help='(str: "./output") Folder name for output simulations',
)
parser.add_argument(
    "--coord-start", type=int)

parser.add_argument("--coord-end", type=int)
parser.add_argument(
    "-s",
    "--sim_id",
    default="run_0",
    help='(str: "run_0") Folder name (under output) for a specific run',
)
parser.add_argument(
    "-a",
    "--continue_sim",
    action="store_true",
    help="Turn on continuation. Files must have same pdb, sim_id, and path as previous run",
)
parser.add_argument(
    "-m",
    "--m_steps",
    type=int,
    default=1000000,
    help="(int: 1e6) Number of Upside timesteps",
)

args = parser.parse_args()


if args.temp is None:
    T = 0.9
else:
    T = args.temp
tempers_str = str(T)

# Add the correct sourcing command
upside_path = os.environ["UPSIDE_HOME"]
upside_utils_dir = os.path.expanduser(upside_path + "/py")
sys.path.insert(0, upside_utils_dir)
import run_upside as ru

# ----------------------------------------------------------------------
## General Settings and Path
# ----------------------------------------------------------------------

pdb_dir = "/project/dinner/scguo/kaiB/dga/seeds_remd"
seed_file = f"{pdb_dir}/files.txt"
is_native = True
ff = "ff_2.1"
duration = args.m_steps
frame_interval = 200  # increase saving interval
randomseed = np.random.randint(0, 100000)
account = "pi-dinner"
# collect coordinate files
with open(seed_file) as f:
    pdb_lines = f.readlines()[args.coord_start:args.coord_end]
    pdb_names = [pdb.strip("\n") for pdb in pdb_lines]
job_names, run_dirs, input_dirs, output_dirs = [], [], [], []
h5_files = []
log_files = []
# run one cis and one trans
sim_ids = ["cis", "trans"]
for coord, idx in zip(pdbs_names, range(args.coord_start, args.coord_end)):
    for sim_id in sim_ids:
        job_name = "{}_{}".format(coord, sim_id)
        job_names.append(job_name)
        output_dir = f"{args.output}/{idx:05}"
        output_dirs.append(output_dir)
        run_dir = "{}/{}".format(output_dir, args.sim_id)
        run_dirs.append(run_dir)
        input_dir = "{}/inputs".format(run_dir)
        input_dirs.append(input_dir)
        h5_files.append("{}/{}.run.up".format(run_dir, coord))
        log_files.append("{}/{}.run.log".format(run_dir, coord))


print(f"{job_names=}")
time_limit = 172000.0
# ----------------------------------------------------------------------
# Set the path and filename
# ADAM EDIT
# Change file structure to have the run directory contain the input directory for that run
# ----------------------------------------------------------------------


make_dirs = [output_dirs, run_dirs, input_dirs]
for direc in make_dirs:
    for d in direc:
        if not os.path.exists(d):
            os.makedir(d)

# ----------------------------------------------------------------------
## Check the previous trajectories if you set continue_sim = True
# ----------------------------------------------------------------------

if args.continue_sim:
    for h5 in h5_files:
        exist = os.path.exists(h5)
        if not exist:
            print("Warning: no previous trajectory file {}!".format(h5))
            print('set "continue_sim = False" and start a new simulation')
            print("Cancelling...")
            sys.exit(2)
    if args.continue_sim:
        exist = os.path.exists(log_file)
        if not exist:
            print("Warning: no previous log file {}!".format(log_file))

# ----------------------------------------------------------------------
## Generate Upside readable initial structure (and fasta) from PDB
# ----------------------------------------------------------------------
# upside_utils_dir = os.environ['UPSIDE_HOME'] + "/py"
if not args.continue_sim:
    for c, input_dir in zip(pdb_names, input_dirs):
        print("Initial structure gen...")
        cmd = (
            "python {0}/PDB_to_initial_structure.py "
            "{1}/{2}.pdb "
            "{3}/{2} "
            "--record-chain-breaks "
            #           "--allow-unexpected-chain-breaks "
        ).format(upside_utils_dir, pdb_dir, c, input_dir)
        print(cmd)

        sp.check_output(cmd.split())

# ----------------------------------------------------------------------
## Configure
# ----------------------------------------------------------------------

# parameters
param_dir_base = os.path.expanduser(upside_path + "/parameters/")
param_dir_common = param_dir_base + "common/"
param_dir_ff = param_dir_base + "{}/".format(ff)

# options
fasta = "{}/{}.fasta".format(input_dir, args.coord)

# ADAM EDIT: Switch the cis-prolines to trans-prolines (or vice versa) using sed
# 0 - cis to trans; 1 - trans to cis; else - none
if args.pro_flag == 0:
    print("Switching prolines (cis to trans)...")
    sp.check_output("sed -i 's/\*PTLAKVL\*PP\*P/PTLAKVLPPP/g' " + fasta, shell=True)
elif args.pro_flag == 1:
    print("Switching prolines (trans to cis)...")
    sp.check_output("sed -i 's/PTLAKVLPPP/\*PTLAKVL\*PP\*P/g' " + fasta, shell=True)
elif args.pro_flag == 3:
    print("Switching P71 (and the others) (trans to cis)...")
    sp.check_output("sed -i 's/PTLAKVLPPP/\*PTLAKVL\*P\*P\*P/g' " + fasta, shell=True)
elif args.pro_flag == 4:
    print("Adding P71 to the cis prolines")
    sp.check_output(
        "sed -i 's/\*PTLAKVL\*PP\*P/\*PTLAKVL\*P\*P\*P/g' " + fasta, shell=True
    )
else:
    print("Do not switch prolines...")

# END EDIT

kwargs = dict(
    rama_library=param_dir_common + "rama.dat",
    rama_sheet_mix_energy=param_dir_ff + "sheet",
    reference_state_rama=param_dir_common + "rama_reference.pkl",
    hbond_energy=param_dir_ff + "hbond.h5",
    rotamer_placement=param_dir_ff + "sidechain.h5",
    dynamic_rotamer_1body=True,
    rotamer_interaction=param_dir_ff + "sidechain.h5",
    environment_potential=param_dir_ff + "environment.h5",
    bb_environment_potential=param_dir_ff + "bb_env.dat",
    chain_break_from_file="{}/{}.chain_breaks".format(input_dir, args.coord),
    trans_cis=args.iso_param,
)

if is_native:
    kwargs["initial_structure"] = "{}/{}.initial.npy".format(input_dir, args.coord)

config_base = "{}/{}.up".format(input_dir, args.coord)

if not args.continue_sim:
    print("Configuring...")
    config_stdout = ru.upside_config(fasta, config_base, **kwargs)
    print("Config commandline options:")
    print(config_stdout)

# ----------------------------------------------------------------------
## Advanced Configure
# ----------------------------------------------------------------------

# Here, we can use the run_upside.advanced_config function to add more advanced configuration
# on the config file obtained in the previous step. advanced_config() allows us to add many
# features, including but not limited to:
#     pulling energy
#     restraints (spring energy)
#     wall potential
#     contact energy


# ----------------------------------------------------------------------
## Run Settings
# ----------------------------------------------------------------------

upside_opts = (
    "--time-limit {} "
    "--duration {} "
    "--frame-interval {} "
    "--temperature {} "
    "--seed {} "
)

upside_opts = upside_opts.format(
    time_limit, duration, frame_interval, tempers_str, randomseed
)


if args.continue_sim:
    print("Archiving prev output...")

    localtime = time.asctime(time.localtime(time.time()))
    localtime = localtime.replace("  ", " ")
    localtime = localtime.replace(" ", "_")
    localtime = localtime.replace(":", "-")

    if os.path.exists(log_file):
        shutil.move(log_file, "{}.bck_{}".format(log_file, localtime))
    else:
        print("Warning: no previous log file {}!".format(log_file))

    for fn in h5_files:
        with tb.open_file(fn, "a") as t:
            i = 0
            while "output_previous_%i" % i in t.root:
                i += 1
            new_name = "output_previous_%i" % i
            if "output" in t.root:
                n = t.root.output
            else:
                n = t.get_node("/output_previous_%i" % (i - 1))

            t.root.input.pos[:, :, 0] = n.pos[-1, 0]

            if "output" in t.root:
                t.root.output._f_rename(new_name)

else:
    for fn in h5_files:
        shutil.copyfile(config_base, fn)

# SLURM options
# Will want to increase the time for production runs


print("Running...")
cmd = "{}/obj/upside {} {}".format(upside_path, upside_opts, h5_files[0])
sp.check_call(cmd, shell=True)
