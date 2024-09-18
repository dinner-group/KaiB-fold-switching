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

BETA = 50
LAMBDA = 1.8


# Modified this to calculate native contacts of a certain section
def best_hummer_q(traj, native, selection, BETA_CONST, LAMBDA_CONST):
    """Compute the fraction of native contacts according the definition from
    Best, Hummer and Eaton [1]

    Parameters
    ----------
    traj : md.Trajectory
        The trajectory to do the computation for
    native : md.Trajectory
        The 'native state'. This can be an entire trajecory, or just a single frame.
        Only the first conformation is used

    Returns
    -------
    q : np.array, shape=(len(traj),)
        The fraction of native contacts in each frame of `traj`

    References
    ----------
    ..[1] Best, Hummer, and Eaton, "Native contacts determine protein folding
          mechanisms in atomistic simulations" PNAS (2013)
    """

    NATIVE_CUTOFF = 0.45  # nanometers

    # get the indices of all of the atoms
    heavy = native.topology.select("resid 0 to 107")
    # get the pairs of heavy atoms which are farther than 3
    # residues apart
    heavy_pairs = np.array(
        [
            (i, j)
            for (i, j) in product(selection, heavy)
            if abs(
                native.topology.atom(i).residue.index
                - native.topology.atom(j).residue.index
            )
            > 3
        ]
    )

    # compute the distances between these pairs in the native state
    heavy_pairs_distances = md.compute_distances(native[0], heavy_pairs)[0]
    # and get the pairs s.t. the distance is less than NATIVE_CUTOFF
    native_contacts = heavy_pairs[heavy_pairs_distances < NATIVE_CUTOFF]
    # print("Number of native contacts", len(native_contacts))

    # now compute these distances for the whole trajectory
    r = md.compute_distances(traj, native_contacts)
    # and recompute them for just the native state
    r0 = md.compute_distances(native[0], native_contacts)

    q = np.mean(1.0 / (1 + np.exp(BETA_CONST * (r - LAMBDA_CONST * r0))), axis=1)
    return q


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

    q_gs_all, q_gs, q_fs_all, q_fs, q_core = [], [], [], [], []
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
        q_frac_gs_all = best_hummer_q(traj, gs_traj, gs_selection, BETA, LAMBDA)
        q_frac_gs = best_hummer_q(traj, gs_traj, gs_switch_selection, BETA, LAMBDA)
        q_frac_fs_all = best_hummer_q(traj, fs_traj, fs_selection, BETA, LAMBDA)
        q_frac_fs = best_hummer_q(traj, fs_traj, fs_switch_selection, BETA, LAMBDA)
        q_frac_core = best_hummer_q(traj, gs_traj, core_selection, BETA, LAMBDA)

        # Accumulate trajs and CVs
        q_gs_all.append(q_frac_gs_all)
        q_gs.append(q_frac_gs)
        q_fs_all.append(q_frac_fs_all)
        q_fs.append(q_frac_fs)
        q_core.append(q_frac_core)

    # Save trajs and CVs
    pickle.dump(q_gs_all, open(f"{input_dir}/{output_dir}/{output_name}_{run_dir}_q_gs_all.pkl", "wb"))
    pickle.dump(q_gs, open(f"{input_dir}/{output_dir}/{output_name}_{run_dir}_q_gs.pkl", "wb"))
    pickle.dump(q_fs_all, open(f"{input_dir}/{output_dir}/{output_name}_{run_dir}_q_fs_all.pkl", "wb"))
    pickle.dump(q_fs, open(f"{input_dir}/{output_dir}/{output_name}_{run_dir}_q_fs.pkl", "wb"))
    pickle.dump(q_core, open(f"{input_dir}/{output_dir}/{output_name}_{run_dir}_q_core.pkl", "wb"))


if __name__ == "__main__":
    main()
