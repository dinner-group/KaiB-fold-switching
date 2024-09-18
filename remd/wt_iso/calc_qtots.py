import os
import sys
import dill as pickle
import numpy as np
import tables as tb
import mdtraj as md
import pyemma

# What I think the parameters are
# 1 --> base directory
# 2 --> output directory name
# 3 --> base name for output files
# 4 --> run directory

# upside_path = '/project/dinner/aanto/upside2'
upside_path = os.environ["UPSIDE_HOME"]
upside_utils_dir = os.path.expanduser(upside_path + "/py")
sys.path.insert(0, upside_utils_dir)

import mdtraj_upside as mu
import glob as glob
from itertools import combinations
from itertools import product
import dill as pickle


# Modified this to calculate native contacts of a certain section
# Also changed lambda from 1.8 to 1.5 - mght be worth testing dif
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


# Modified this to calculate native contacts of a certain section
# Also changed lambda from 1.8 to 1.5 - mght be worth testing dif
def best_hummer_qtot(traj, ref, selection1, selection2):
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

    BETA_CONST = 50  # 1/nm
    LAMBDA_CONST = 1.8
    NATIVE_CUTOFF = 0.45  # nanometers

    # get the pairs of heavy atoms which are farther than 3
    # residues apart
    heavy_pairs = np.array(
        [
            (i, j)
            for (i, j) in product(selection1, selection2)
            if abs(
                traj[0].topology.atom(i).residue.index
                - traj[0].topology.atom(j).residue.index
            )
            > 3
        ]
    )

    # now compute these distances for the whole trajectory
    r = md.compute_distances(traj, heavy_pairs)
    r0 = md.compute_distances(ref[0], heavy_pairs)
    q0 = np.sum(1.0 / (1 + np.exp(BETA_CONST * (r0 - LAMBDA_CONST * NATIVE_CUTOFF))))
    q = np.sum(
        1.0 / (1 + np.exp(BETA_CONST * (r - LAMBDA_CONST * NATIVE_CUTOFF))), axis=1
    )
    return q / q0


def moving_average(a, n=2):
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    padding = np.tile(ret[n - 1] / n, (n - 1, 1))
    avged = np.vstack((padding, ret[n - 1 :] / n))
    return avged


def main():
    input_dir = sys.argv[1]
    run_dir = sys.argv[4]
    dfiles = sorted(glob.glob("{}/{}/*.up".format(input_dir, run_dir)))
    output_dir = sys.argv[2]
    output_name = sys.argv[3]

    # Make differenct secondary strucutre selections for gs nd fs KaiB
    stride = 1
    backbone = "((name N or name CA or name C or name O) and (resid 0 to 105))"
    gs_file = "/project/dinner/scguo/kaiB/remd/wt_iso/gs/2qke_mon.run.00.up"
    t_gs = mu.load_upside_traj(gs_file, stride=stride)
    gs_top = t_gs.top
    gs_red = gs_top.select("{} and (resid 8 to 13)".format(backbone))
    gs_1 = gs_top.select("{} and (resid 58 to 60)".format(backbone)) # domain boundaries from inspecting PDB annotation
    gs_orange = gs_top.select("{} and (resid 61 to 66)".format(backbone))
    gs_green = gs_top.select("{} and (resid 71 to 80)".format(backbone))
    gs_blue = gs_top.select("{} and (resid 87 to 94)".format(backbone))
    gs_selections = [gs_1, gs_orange, gs_green, gs_blue]
    gs_specials = [gs_blue, None, None, gs_red]

    fs_file = "/project/dinner/scguo/kaiB/remd/wt_iso/fs/fs_wtseq.run.00.up"
    t_fs = mu.load_upside_traj(fs_file, stride=stride)
    fs_top = t_fs.top
    fs_red = fs_top.select("{} and (resid 8 to 13)".format(backbone))
    fs_1 = fs_top.select("{} and (resid 49 to 56)".format(backbone))
    fs_orange = fs_top.select("{} and (resid 63 to 68)".format(backbone))
    fs_green = fs_top.select("{} and (resid 72 to 76)".format(backbone))
    fs_blue = fs_top.select("{} and (resid 83 to 100)".format(backbone))
    fs_selections = [fs_1, fs_orange, fs_green, fs_blue]
    fs_specials = [None, fs_red, fs_orange, None]

    red_selection = "(resid 8 to 13 or resid 20 to 34 or resid 40 to 45)"
    selection = "(resid 8 to 13 or resid 20 to 34 or resid 40 to 45 or resid 61 to 66 or resid 71 to 80 or resid 87 to 94)"
    fs2_selection = "(resid 8 to 13 or resid 20 to 34 or resid 40 to 45 or resid 63 to 68 or resid 72 to 76 or resid 83 to 100)"
    refs = gs_top.select("{} and {}".format(backbone, selection))
    red_refs = gs_top.select("{} and {}".format(backbone, red_selection))
    fs_refs = fs_top.select("{} and {}".format(backbone, fs2_selection))

    # 406 dimensional basis set, using pyemma
    feat = pyemma.coordinates.featurizer(
        fs_top
    )  # /project/dinner/aanto/kaiB/up2/dga_pdbs/s00i00.pdb')
    nterm_sel = "(resid 8 to 13 or resid 20 to 34 or resid 40 to 45)"
    cterm_sel = "(resid 61 to 66 or resid 71 to 80 or resid 86 to 93)"
    feat_ref = fs_top.select("(name CA) and ({} or{})".format(nterm_sel, cterm_sel))
    feat.add_distances(feat_ref[::2], periodic=False)

    fs_qtots = []
    write_dir = f"{input_dir}/{output_dir}"
    print("{}/{}/*.up".format(input_dir, run_dir), dfiles)
    print(f"Output directory name: {output_dir}")
    print(f"Output base name: {output_name}")
    print(f"Writing output files to {write_dir}")
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)

    # Loop through all the data files
    for i, dfile in enumerate(dfiles):
        print(dfile)
        # Load and save in mdtraj format
        h5_file = "{}.h5".format(dfile[:-3])
        t = mu.load_upside_traj(dfile, stride=stride)
        top1 = t.top
        red_refs1 = top1.select("{} and {}".format(backbone, red_selection))
        refs1 = top1.select("{} and {}".format(backbone, selection))
        fs_refs1 = top1.select("{} and {}".format(backbone, fs2_selection))
        t.superpose(reference=t_gs, atom_indices=red_refs1, ref_atom_indices=red_refs)

        # Calculate the secondary structure CVs
        qs = []
        for gs_selection, gs_special, fs_selection, fs_special in zip(
            gs_selections, gs_specials, fs_selections, fs_specials
        ):
            if fs_special is not None:
                qs.append(best_hummer_qtot(t, t_fs, fs_selection, fs_special))
            else:
                qs.append(best_hummer_q(t, t_fs, fs_selection, 50, 1.8))
            if gs_special is not None:
                qs.append(best_hummer_qtot(t, t_gs, gs_selection, gs_special))
            else:
                qs.append(best_hummer_q(t, t_gs, gs_selection, 50, 1.8))
        qs = np.asarray(qs)
        smoothed_qs = moving_average(qs.T, n=3)

        # Accumulate trajs and CVs
        fs_qtots.append(qs)
    # Save trajs and CVs
    # pickle.dump(smoothed_fs_qtots, open('{}/{}_smoothed_fs_qtots_{}.pkl'.format(output_dir, iso, j), 'wb'))
    pickle.dump(fs_qtots, open(f"{input_dir}/{output_dir}/{output_name}_{run_dir}_fs_qtots.pkl", "wb"))

if __name__ == "__main__":
    main()
