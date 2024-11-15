import sys, os, shutil
import subprocess as sp
import numpy as np
import tables as tb
from math import sqrt
import time
import argparse

upside_path = os.environ['UPSIDE_HOME']
upside_utils_dir = os.path.expanduser(upside_path+"/py")
sys.path.insert(0, upside_utils_dir)
import run_upside as ru

#----------------------------------------------------------------------
## Added to enable general parsing
#----------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Convert a PDB to an Upside topology file appropriate for performing PS calculations")
parser.add_argument(
    'input',
    help="Input PDB name without directory name"
)
parser.add_argument(
    "--pdb-dir",
    help="Directory PDB is in"
)
parser.add_argument(
    "-d",
    "--dir",
    default="./",
    help="Work directory"
)
args = parser.parse_args()


pdb_id           = args.input
pdb_dir          = args.pdb_dir
is_native        = True
ff               = 'ff_2.1'
work_dir         = args.dir
input_dir        = "{}/inputs".format(work_dir)

#----------------------------------------------------------------------
## Configure
#----------------------------------------------------------------------

# parameters
param_dir_base = os.path.expanduser(upside_path+"/parameters/")
param_dir_common = param_dir_base + "common/"
param_dir_ff = param_dir_base + '{}/'.format(ff)

# options
fasta = "{}/{}.fasta".format(input_dir, pdb_id)
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
               chain_break_from_file     = "{}/{}.chain_breaks".format(input_dir, pdb_id),
               use_heavy_atom_coverage   = True
             )

if is_native:
    kwargs['initial_structure'] =  "{}/{}.initial.npy".format(input_dir, pdb_id)

config_base = "{}/{}-HDX.up".format( input_dir, pdb_id)

print ("Configuring...")
config_stdout = ru.upside_config(fasta, config_base, **kwargs)
print ("Config commandline options:")
print (config_stdout)
