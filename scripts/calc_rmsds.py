import os
import sys
import dill as pickle
import numpy as np
import tables as tb
import mdtraj as md
import pyemma

upside_path = os.environ["UPSIDE_HOME"]
upside_utils_dir = os.path.expanduser(upside_path + "/py")
sys.path.insert(0, upside_utils_dir)

import mdtraj_upside as mu
import glob as glob
from itertools import combinations
from itertools import product
import dill as pickle


def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    output_name = sys.argv[3]
    run_dir = sys.argv[4]
    dfiles = sorted(glob.glob("{}/{}/*.up".format(input_dir, run_dir)))

    # Make differenct secondary strucutre selections for gs nd fs KaiB
    stride = 1
    gs_file = "/project2/dinner/darrenjl/kaiB/03022023_restart_trans_gs/run_0/2qke_mutated.run.00.up"
    gs_traj = mu.load_upside_traj(gs_file)[0]
    fs_file = "/project2/dinner/darrenjl/kaiB/02272023_cis_fs/run_0/fs_mutated.run.00.up"
    fs_traj = mu.load_upside_traj(fs_file)[0]

    rmsd_gs_all, rmsd_gs, rmsd_fs_all, rmsd_fs, rmsd_core = [], [], [], [], []
    # first 51 residues
    core_selection = gs_traj.top.select("protein and resid 0 to 50")
    gs_selection = gs_traj.top.select("protein")
    gs_switch_selection = gs_traj.top.select("protein and resid 51 to 94")
    fs_selection = fs_traj.top.select("protein")
    fs_switch_selection = fs_traj.top.select("protein and resid 51 to 94")

    write_dir = f"{input_dir}/{output_dir}"
    print("{}/{}/*.up".format(input_dir, run_dir), dfiles)
    print(f"Output directory name: {output_dir}")
    print(f"Output base name: {output_name}")
    print(f"Writing output files to {write_dir}")
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)

    # Loop through all the data files
    for i, dfile in enumerate(dfiles):
        print(f"Analyzing {dfile}")
        # Load
        h5_file = "{}.h5".format(dfile[:-3])
        traj = md.load(h5_file)

        # Calculate the contact fractions
        rmsd_gs_all.append(md.rmsd(traj, gs_traj, atom_indices=gs_selection))
        rmsd_fs_all.append(md.rmsd(traj, fs_traj, atom_indices=fs_selection))
        rmsd_gs.append(md.rmsd(traj, gs_traj, atom_indices=gs_switch_selection))
        rmsd_fs.append(md.rmsd(traj, fs_traj, atom_indices=fs_switch_selection))
        rmsd_core.append(md.rmsd(traj, gs_traj, atom_indices=core_selection))

    # Save trajs and CVs
    np.save(f"{input_dir}/{output_dir}/{output_name}_{run_dir}_rmsd_gs_all.npy", rmsd_gs_all)
    np.save(f"{input_dir}/{output_dir}/{output_name}_{run_dir}_rmsd_fs_all.npy", rmsd_fs_all)
    np.save(f"{input_dir}/{output_dir}/{output_name}_{run_dir}_rmsd_gs.npy", rmsd_gs)
    np.save(f"{input_dir}/{output_dir}/{output_name}_{run_dir}_rmsd_fs.npy", rmsd_fs)
    np.save(f"{input_dir}/{output_dir}/{output_name}_{run_dir}_rmsd_core.npy", rmsd_core)

if __name__ == "__main__":
    main()
