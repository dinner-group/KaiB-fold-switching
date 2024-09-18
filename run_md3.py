import numpy as np
import sys, os, shutil
import subprocess as sp
import numpy as np
import tables as tb
from math import sqrt
import time

## ADAM EDIT: Add an argument parser to allow for flexible run-time settings from the command line ##
import argparse
parser = argparse.ArgumentParser(description="Run an Upside (Set of) Simulation(s)")
parser.add_argument("-d", "--dinner", action="store_true",
                    help="Will use the dinner partition instead of caslake")
parser.add_argument("-b", "--beagle", action="store_true",
                    help="Will use the beagle partition instead of caslake")
parser.add_argument("-l", "--long", action="store_true",
                    help="Will use the beagle-long partition instead of caslake")
parser.add_argument("-n", "--n_rep", type=int, default=48,
                    help="(int: 48) Number of replicas (one per CPU core), between 1 and 48")
parser.add_argument("-r", "--res_k", default=None, type=float,
                    help="(float: None) If given, model the insulin cystein bonds with spring potentials. Give k (suggested 4.0).")
parser.add_argument("-p", "--pro_flag", type=int, default=2,
        help="(int: 2) Flag to determine if to swtich proline isomerization state. 0,1, or 2. 2: Won't edit prolines. 0: fs kaiB (cis to trans), 1: gs kaiB (trans to cis), 3: gs --> fs kaiB except where P71 is included in accordance with Adams sims), 4: fs --> fs kaiB except P71 is included")
parser.add_argument("-i", "--iso_param", default='',
                    help="(str: \'\') If applicable, paramteter file for the proline isomerization")
parser.add_argument("-t", "--temp", default=None,
                    help="(float: None) Temperature for a constant T simulation. If not specified, revert to melting T ramp")
parser.add_argument("-o", "--output", default="./output",
                    help="(str: \"./output\") Folder name for output simulations")
parser.add_argument("-c", "--coord", default='',
                    help="(str: \'\') Path to pdb file. Omit the .pdb ending")
parser.add_argument("-s", "--sim_id", default="run_0",
                    help="(str: \"run_0\") Folder name (under output) for a specific run")
parser.add_argument("-e", "--ex_int", type=int, default=None,
                    help="(int: None) If applicable, replica exchange interval in units of upside timesteps. Suggested 10")
parser.add_argument("-a", "--continue_sim", action="store_true",
                    help="Turn on continuation. Files must have same pdb, sim_id, and path as previous run")
parser.add_argument("-m", "--m_steps", type=int, default=1000000,
                    help="(int: 1e6) Number of Upside timesteps")

args = parser.parse_args()

if args.dinner==True:
    partition='dinner'              # use dinner if arg passed 
    time_limit=172000.0
    run_time='48:00:00'
    mem='8G'
    #qos='weare-dinner'
elif args.beagle==True:
    partition='beagle3'              # use beagle if arg passed 
    time_limit=172000.0
    run_time='48:00:00'
    mem='8G'
    qos='beagle3'
elif args.long==True:
    partition='beagle3'              # use beagle-long if arg passed 
    time_limit=259000.0
    run_time='72:00:00'
    #mem='250G'
    mem='60G'
    qos='beagle3-long'
else:
    partition='caslake'             # use caslake by default
    time_limit=129000.0              
    run_time='36:00:00'     
    mem='180G'
    qos='caslake'

if args.res_k is not None:
    res_grps = ["5,10", "6,27", "19,39"]     # For ins monomer
    #res_grps = ["6,42", "18,55", "41,46"]   # For SCI-c
else:
    res_grps = []

if args.temp is None:
    T_low = 0.78 
    T_high = 1.02
    tempers =  np.linspace(T_low**0.5, T_high**0.5, args.n_rep)**2
    tempers_str = ",".join(str(t) for t in tempers)
else:
    tempers = np.repeat(args.temp, args.n_rep)
    tempers_str = ",".join(str(t) for t in tempers)

replica_interval=args.ex_int
if replica_interval is not None:
    exchange=True
else:
    exchange=False

##END ADAM EDIT

## ADAM EDIT ##
# Add the correct sourcing command
#sp.check_output("source /project/dinner/aanto/upside2/source.sh", shell=True)
## END ADAM EDIT ##
upside_path = os.environ["UPSIDE_HOME"]
upside_utils_dir = os.path.expanduser(upside_path + "/py")
sys.path.insert(0, upside_utils_dir)
import run_upside as ru

#----------------------------------------------------------------------
## General Settings and Path
#----------------------------------------------------------------------

pdb_dir          = '/project/dinner/scguo/kaiB/dga/seeds/'#'./pdb'
is_native        = True
ff               = 'ff_2.1'
#duration         = 2000000
duration         = args.m_steps
frame_interval   = 50
randomseed       =  np.random.randint(0,100000) 
account          = "pi-dinner"
job_name         = '{}_{}'.format(args.coord, args.sim_id)

#----------------------------------------------------------------------
# Set the path and filename
# ADAM EDIT
# Change file structure to have the run directory contain the input directory for that run
#----------------------------------------------------------------------

output_dir = args.output
run_dir    = "{}/{}".format(output_dir, args.sim_id)
input_dir  = "{}/inputs".format(run_dir)

make_dirs = [output_dir, run_dir, input_dir]
for direc in make_dirs:
    if not os.path.exists(direc):
        os.makedirs(direc)

h5_files = []
for j in range(args.n_rep):
    h5_file  = "{}/{}.run.{}.up".format(run_dir, args.coord, str(j).zfill(2))
    h5_files.append(h5_file)
h5_files_str = " ".join(h5 for h5 in h5_files)
log_file = "{}/{}.run.log".format(run_dir, args.coord)

#----------------------------------------------------------------------
## Check the previous trajectories if you set continue_sim = True 
#----------------------------------------------------------------------

if args.continue_sim:
    for h5 in h5_files:
        exist = os.path.exists(h5)
        if not exist:
            print('Warning: no previous trajectory file {}!'.format(h5))
            print('set "continue_sim = False" and start a new simulation')
            print('Cancelling...')
            sys.exit(2)
    if args.continue_sim:
        exist = os.path.exists(log_file)
        if not exist:
            print('Warning: no previous log file {}!'.format(log_file))

#----------------------------------------------------------------------
## Generate Upside readable initial structure (and fasta) from PDB 
#----------------------------------------------------------------------
# upside_utils_dir = os.environ['UPSIDE_HOME'] + "/py"
if not args.continue_sim:
    print ("Initial structure gen...")
    cmd = (
           "python {0}/PDB_to_initial_structure.py "
           "{1}/{2}.pdb "
           "{3}/{2} "
           "--record-chain-breaks "
#           "--allow-unexpected-chain-breaks "
          ).format(upside_utils_dir, pdb_dir, args.coord, input_dir )
    print (cmd)

    sp.check_output(cmd.split())

#----------------------------------------------------------------------
## Configure
#----------------------------------------------------------------------

# parameters
param_dir_base = os.path.expanduser(upside_path+"/parameters/")
param_dir_common = param_dir_base + "common/"
param_dir_ff = param_dir_base + '{}/'.format(ff)

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
    sp.check_output("sed -i 's/\*PTLAKVL\*PP\*P/\*PTLAKVL\*P\*P\*P/g' " + fasta, shell=True)
else:
    print("Do not switch prolines...")

# END EDIT

kwargs = dict(
               rama_library              = param_dir_common + "rama.dat",
               rama_sheet_mix_energy     = param_dir_ff + "sheet",
               reference_state_rama      = param_dir_common + "rama_reference.pkl",
               hbond_energy              = param_dir_ff + "hbond.h5",
               rotamer_placement         = param_dir_ff + "sidechain.h5",
               dynamic_rotamer_1body     = True,
               rotamer_interaction       = param_dir_ff + "sidechain.h5",
               environment_potential     = param_dir_ff + "environment.h5",
               bb_environment_potential  = param_dir_ff + "bb_env.dat",
               chain_break_from_file     = "{}/{}.chain_breaks".format(input_dir, args.coord),
               trans_cis                 = args.iso_param
             )

if is_native:
    kwargs['initial_structure'] =  "{}/{}.initial.npy".format(input_dir, args.coord)

config_base = "{}/{}.up".format( input_dir, args.coord)

if not args.continue_sim:
    print ("Configuring...")
    config_stdout = ru.upside_config(fasta, config_base, **kwargs)
    print ("Config commandline options:")
    print (config_stdout)

#----------------------------------------------------------------------
## Advanced Configure
#----------------------------------------------------------------------

# Here, we can use the run_upside.advanced_config function to add more advanced configuration
# on the config file obtained in the previous step. advanced_config() allows us to add many
# features, including but not limited to:
#     pulling energy
#     restraints (spring energy)
#     wall potential
#     contact energy

if args.res_k is not None:
    kwargs = dict(
               restraint_groups = res_grps,
               restraint_spring_constant = res_k
               #fixed_wall = 'wall-const-xyz.dat'
               #pair_wall = 'wall-pair-xyz.dat'
               #fixed_spring = 'spring-const-xyz.dat'
               #pair_spring = 'spring-pair-xyz.dat'
               #nail = 'nail-xyz.dat'
        )

    config_stdout = ru.advanced_config(config_base, **kwargs)
    print ("Advanced Config commandline options:")
    print (config_stdout)


#----------------------------------------------------------------------
## Run Settings
#----------------------------------------------------------------------

upside_opts = (
                 "--time-limit {} "
                 "--duration {} "
                 "--frame-interval {} "
                 "--temperature {} "
                 "--seed {} "
               )

if exchange:
    swap_sets    = ru.swap_table2d(1, len(tempers)) # specifies which replicas are able to exchange 
    upside_opts += "--replica-interval {} --swap-set {} --swap-set {} " # only perform swaps for replex; duration of time until swap is attempted
    upside_opts  = upside_opts.format(time_limit, duration, frame_interval, tempers_str, randomseed, replica_interval, swap_sets[0], swap_sets[1])
else:
    upside_opts  = upside_opts.format(time_limit, duration, frame_interval, tempers_str, randomseed)


if args.continue_sim:
    print ("Archiving prev output...")

    localtime = time.asctime( time.localtime(time.time()) )
    localtime = localtime.replace('  ', ' ')
    localtime = localtime.replace(' ', '_')
    localtime = localtime.replace(':', '-')

    if os.path.exists(log_file):
        shutil.move(log_file, '{}.bck_{}'.format(log_file, localtime))
    else:
        print('Warning: no previous log file {}!'.format(log_file))

    for fn in h5_files:
        with tb.open_file(fn, 'a') as t:
            i = 0
            while 'output_previous_%i'%i in t.root:
                i += 1
            new_name = 'output_previous_%i'%i
            if 'output' in t.root:
                n = t.root.output
            else:
                n = t.get_node('/output_previous_%i'%(i-1))

            t.root.input.pos[:,:,0] = n.pos[-1,0]

            if 'output' in t.root:
                t.root.output._f_rename(new_name)

else:
    for fn in h5_files:
        shutil.copyfile(config_base, fn)

# SLURM options
# Will want to increase the time for production runs 

#If you run on beagle, account for the fact that they only allow for 36 processors for a node
if args.beagle == True:
    sbatch_opts = (
                    "--account={} " 
                    "--job-name={} "
                    "--output={} "
                    "--time={} "
                    "--partition={} "
                    "--nodes=1 "
                    "--ntasks-per-node={} "
                    "--mem={} "
                  )
    sbatch_opts = sbatch_opts.format(account, job_name, log_file, run_time, partition, args.n_rep, mem) 

else:
   sbatch_opts = (
                "--account={} " 
                "--job-name={} "
                "--output={} "
                "--time={} "
                "--partition={} "
                "--nodes=1 "
                "--ntasks-per-node={} "
                "--mem={} "
              )
   sbatch_opts = sbatch_opts.format(account, job_name, log_file, run_time, partition, args.n_rep, mem) 


print ("Running...")
cmd = "sbatch {} --wrap=\"{}/obj/upside {} {}\"".format(sbatch_opts, upside_path, upside_opts, h5_files_str)
sp.check_call(cmd, shell=True)





