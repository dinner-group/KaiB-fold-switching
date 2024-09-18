import os
import sys
import dill
import logging
import argparse

import numpy as np
import scipy
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

import extq

upside_path = "/project/dinner/scguo/upside2"
upside_utils_dir = os.path.expanduser(upside_path + "/py")
sys.path.insert(0, upside_utils_dir)

base_dir = ""

def load_cvs(base_dir, n_s, n_i):
    raw_feats, fs_qtots, f_rmsds, p_rmsds, r_rmsds = [], [], [], [], []
    for i in range(n_s):
        for j in range(n_i):
            for iso in ("cis", "trans"):
                idx = f"{i:02}_{j:02}_{iso}"
                head = f"{idx}_dga"
                if not os.path.exists(f"{base_dir}/{idx}/outputs/{head}_raw_feats.pkl"):
                    continue
                raw_feats.extend(
                    np.load(
                        f"{base_dir}/{idx}/outputs/{head}_raw_feats.pkl",
                        allow_pickle=True,
                    )
                )
                fs_qtots.extend(
                    np.load(
                        f"{base_dir}/{idx}/outputs/{head}_fs_qtots.pkl",
                        allow_pickle=True,
                    )
                )
                f_rmsds.extend(
                    np.load(
                        f"{base_dir}/{idx}/outputs/{head}_f_rmsds.pkl",
                        allow_pickle=True,
                    )
                )
                p_rmsds.extend(
                    np.load(
                        f"{base_dir}/{idx}/outputs/{head}_p_rmsds.pkl",
                        allow_pickle=True,
                    )
                )
                r_rmsds.extend(
                    np.load(
                        f"{base_dir}/{idx}/outputs/{head}_r_rmsds.pkl",
                        allow_pickle=True,
                    )
                )
    return raw_feats, fs_qtots, f_rmsds, p_rmsds, r_rmsds


def split_indices(arrays):
    """Gets the indices for np.split from a
    list of arrays.

    Parameters
    ----------
    arrays : ndarray or list/tuple of ndarray
        Arrays from which to get indices

    Returns
    -------
    traj_inds : list of int
        Frame separators to use in np.split
    """
    traj_lens = [len(traj) for traj in arrays]
    traj_inds = []
    subtot = 0
    for length in traj_lens[:-1]:
        subtot += length
        traj_inds.append(subtot)
    return traj_lens, traj_inds

def lag_weights(weights, lag):
    result = []
    for w in weights:
        w[len(w) - lag :] = 0
        result.append(w)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("temp")
    parser.add_argument("-w", "--weights", action='store_true')
    args = parser.parse_args()
    home_dir = "/project/dinner/scguo/kaiB"
    global base_dir
    base_dir = f"/project/dinner/scguo/kaiB/dga/{args.temp}"

    weights_precomputed = args.weights

    logging.basicConfig(
        filename=f"dga_{args.temp}.log",
        encoding="utf-8",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )

    logging.info("Loading features and CVs")
    raw_feats, fs_qtots, f_rmsds, p_rmsds, r_rmsds = load_cvs(base_dir, 7, 32)
    logging.info("Finished loading.")

    c_gsa1 = np.concatenate([traj[1, :] for traj in fs_qtots])
    c_gsa2 = np.concatenate([traj[3, :] for traj in fs_qtots])
    c_gsb2 = np.concatenate([traj[5, :] for traj in fs_qtots])

    c_fsb1 = np.concatenate([traj[0, :] for traj in fs_qtots])
    c_fsb2 = np.concatenate([traj[2, :] for traj in fs_qtots])
    c_fsa2 = np.concatenate([traj[4, :] for traj in fs_qtots])

    c_green_arr = c_gsa2 - c_fsb2
    c_blue_arr = c_gsb2 - c_fsa2
    c_orange_arr = c_gsa1 - c_fsb1

    p_rmsd_arr = np.asarray(p_rmsds).ravel()
    f_rmsd_arr = np.asarray(f_rmsds).ravel()
    in_fs = np.logical_and(
        c_green_arr < -0.78,
        np.logical_and(
            c_blue_arr < -0.83, np.logical_and(c_orange_arr < -0.75, f_rmsd_arr < 0.35)
        ),
    )

    in_gs = np.logical_and(
        c_green_arr > 0.67,
        np.logical_and(
            c_blue_arr > 0.88, np.logical_and(c_orange_arr > 0.75, p_rmsd_arr < 0.45)
        ),
    )
    traj_lens, traj_inds = split_indices(raw_feats)
    c_green_trajs = np.split(c_green_arr, traj_inds)
    c_blue_trajs = np.split(c_blue_arr, traj_inds)
    c_orange_trajs = np.split(c_orange_arr, traj_inds)

    if not weights_precomputed:
        raise ValueError("Must specify file with weights")
    else:
        logging.info("Loading precomputed weights from %s", f"{base_dir}/dga_data/weights.npy")
        weights = np.load(f"{base_dir}/dga_data/weights.npy", allow_pickle=True)[-1]

    logging.info("Loading committors...")
    in_domain = ~(in_fs | in_gs)
    in_d = np.split(in_domain, traj_inds)
    # use  committors corresponding to lag = 2000
    qp_fs2gs = np.load(f"{base_dir}/dga_data/qp_fs2gs.npy", allow_pickle=True)
    qp_gs2fs = np.load(f"{base_dir}/dga_data/qp_gs2fs.npy", allow_pickle=True)
    qm_fs2gs = np.load(f"{base_dir}/dga_data/qm_fs2gs.npy", allow_pickle=True)
    qm_gs2fs = np.load(f"{base_dir}/dga_data/qm_gs2fs.npy", allow_pickle=True)

    logging.info("Computing currents...")
    current_lags = np.array(
            [100, 200, 500, 1000, 2000, 5000], dtype=int
    )
    logging.info("fs->gs direction")
    # cvs = [qp_fs2gs, r_rmsds, c_green_trajs, c_blue_trajs, c_orange_trajs]
    # names = ['qp', 'r_rmsds', 'green', 'blue', 'orange']
    # for cv, name in zip(cvs, names):
    #     j_lags = []
    #     for lag in current_lags:
    #         current = extq.tpt.current(qp_fs2gs, qm_fs2gs, weights, in_d, cv, lag)
    #         j_lags.append(current)
    #     np.save(f"{base_dir}/dga_data/j_fs2gs_{name}.npy", j_lags)

    committor_lags = current_lags
    # cvs = [qp_fs2gs[4], r_rmsds, c_green_trajs, c_blue_trajs, c_orange_trajs]
    # cvs = [c_blue_trajs, c_orange_trajs]
    # names = ['blue', 'orange']
    names = ['qp', 'r_rmsds', 'green', 'blue', 'orange']
    # for cv, name in zip(cvs, names):
    #     for qlag, qp, qm in zip(committor_lags, qp_fs2gs, qm_fs2gs):
    #         j_lags = []
    #         for lag in current_lags:
    #             logging.info("Computing currents along CV %s using committors with lag %s using lag time %s", name, qlag, lag)
    #             current = extq.tpt.current(qp, qm, weights, in_d, cv, lag)
    #         j_lags.append(current)
    #         np.save(f"{base_dir}/dga_data/j_fs2gs_qlag{qlag}_{name}.npy", j_lags)


    logging.info("gs->fs direction")
    cvs = [qp_gs2fs[4], r_rmsds, c_green_trajs, c_blue_trajs, c_orange_trajs]
    for cv, name in zip(cvs, names):
        j_lags = []
        for lag in current_lags:
            current = extq.tpt.current(qp_gs2fs[4], qm_gs2fs[4], weights, in_d, cv, lag)
            j_lags.append(current)
        np.save(f"{base_dir}/dga_data/j_gs2fs_{name}.npy", j_lags)


if __name__ == "__main__":
    main()
