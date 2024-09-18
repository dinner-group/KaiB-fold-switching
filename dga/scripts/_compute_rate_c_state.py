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
# import ivac

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


def make_sparse_basis(dtrajs):
    """Converts a discretized trajectory (e.g. from k-means clustering)
    into a sparse basis of indicator functions.

    Parameters
    ----------
    dtrajs : ndarray
        discretized trajectories

    Return
    ------
    basis : scipy.sparse.csr_matrix
    """
    nclusters = len(np.unique(dtrajs))
    rows, cols = [], []
    for i in range(nclusters):
        pts = np.argwhere(dtrajs == i)
        # indices of which frames are in the cluster i
        rows.append(pts.squeeze())
        # all assigned as 1 in the basis
        cols.append(np.repeat(i, len(pts)))
    rows = np.hstack(rows)
    cols = np.hstack(cols)
    data = np.ones(len(rows), dtype=float)
    basis = scipy.sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(dtrajs), nclusters)
    )
    return basis

def make_dense_basis(dtrajs):
    """Converts a discretized trajectory (e.g. from k-means clustering)
    into a sparse basis of indicator functions.

    Parameters
    ----------
    dtrajs : np.ndarray
    discretized trajectories

    Return
    ------
    basis : np.ndarray
    """

    n_basis = len(np.unique(dtrajs))
    basis = np.zeros((len(dtrajs), n_basis))
    basis[np.arange(len(dtrajs)), dtrajs] += 1.0
    return basis


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


def _basis_helper(dtraj, in_dc):
    # basis_arr = make_sparse_basis(dtraj)[:, :-1]
    basis_arr = make_dense_basis(dtraj)[:, :-1]
    basis_arr[in_dc] = 0
    return scipy.sparse.csr_matrix(basis_arr)


def make_basis(dtraj, in_fs, in_gs, in_c, traj_lens, traj_inds):
    basis_d_arr = _basis_helper(dtraj, (in_fs | in_gs))
    basis_d_withc_arr = _basis_helper(dtraj, (in_fs | in_gs | in_c))

    basis_d, basis_d_withc = [], []
    curr = 0
    for t_len in traj_lens:
        basis_d.append(basis_d_arr[curr : curr + t_len])
        basis_d_withc.append(basis_d_withc_arr[curr : curr + t_len])
        curr += t_len

    logging.info("Length of basis %s, basis shape %s", len(basis_d), basis_d[0].shape)
    return basis_d, basis_d_withc


def lag_weights(weights, lag):
    result = []
    for w in weights:
        w[len(w) - lag :] = 0
        result.append(w)
    return result


def tpt_rate(rate_lags, traj_inds, in_A, in_B, qp, qm, weights, in_d):
    in_Ac = np.split((~in_A).astype(float), traj_inds)
    in_B = np.split(in_B.astype(float), traj_inds)
    qm_rc = [(1 - q) for q in qm]
    n_lags = len(rate_lags)
    rates_lags = np.zeros((4, n_lags))
    rcs = [qp, qm_rc, in_B, in_Ac]

    for j, rc in enumerate(rcs):
        for k, lag in enumerate(rate_lags):
            logging.debug(
                "Computing rates for committor lag %s, and RC %s rate lag %s...",
                j,
                lag,
            )
            com = lag_weights(weights[-1], lag)
            rates_lags[j, k] = extq.tpt.rate(qp, qm, com, in_d, rc, lag)
    return rates_lags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("temp")
    parser.add_argument("--ivac", action='store_true')
    parser.add_argument("--km", action='store_true')
    parser.add_argument("-w", "--weights", action='store_true')
    parser.add_argument("-q", "--committors", action='store_true')
    args = parser.parse_args()
    home_dir = "/project/dinner/scguo/kaiB"
    global base_dir
    base_dir = f"/project/dinner/scguo/kaiB/dga/{args.temp}"

    ivac_precomputed = args.ivac
    km_precomputed = args.km
    weights_precomputed = args.weights
    committors_precomputed = args.committors

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
    r_rmsd_arr = np.asarray(r_rmsds).ravel()
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
    in_c = (r_rmsd_arr > 0.7)

    if not ivac_precomputed:
        logging.info("Running IVAC...")
        ivac_trajs = run_ivac(raw_feats)
    else:
        logging.info("Loading IVAC data from %s", f"{base_dir}/dga_data/ivac_10d_10-1000.npy") 
        ivac_trajs = np.load(f"{base_dir}/dga_data/ivac_10d_10-1000.npy")
    if not km_precomputed:
        logging.info("Clustering...")
        dtraj = cluster(ivac_trajs)
    else:
        logging.info("Loading clustered data from %s", f"{base_dir}/dga_data/dtraj_300.npy")
        dtraj = np.load(f"{base_dir}/dga_data/dtraj_300.npy").squeeze()

    traj_lens, traj_inds = split_indices(raw_feats)
    

    lags = np.array([100, 200, 500, 1000, 2000, 5000], dtype=int)
    if not weights_precomputed:
        logging.info("Computing weights...")
        weights = []
        for lag in lags:
            weights.append(extq.dga.reweight(basis, lag))
        logging.info("Saving weights to %s", f"{base_dir}/dga_data/weights.npy")
        np.save(f"{base_dir}/dga_data/weights.npy", weights, allow_pickle=True)
    else:
        logging.info("Loading precomputed weights from %s", f"{base_dir}/dga_data/weights.npy")
        weights = np.load(f"{base_dir}/dga_data/weights.npy", allow_pickle=True)
    in_dc_withc = (in_fs | in_gs | in_c)
    guess_fs = np.split(in_fs.astype(float), traj_inds)
    guess_gs = np.split(in_gs.astype(float), traj_inds)
    guess_fs_withc = np.split((in_fs + in_c).astype(float), traj_inds)
    guess_gs_withc = np.split((in_gs + in_c).astype(float), traj_inds)
    in_domain = ~(in_fs | in_gs)
    in_d = np.split(in_domain, traj_inds)
    in_d_withc = np.split(~in_dc_withc, traj_inds)

    lag = 1000
    mem = 4
    w = weights[4]

    if not committors_precomputed:
        logging.info("Making basis...")

        basis_d, basis_d_withc = make_basis(
            dtraj, in_fs, in_gs, in_c, traj_lens, traj_inds
        )
        logging.info("Computing committors...")
        qp_fs2gs = extq.memory.forward_committor(basis_d, w, in_d, guess_gs, lag, mem)
        qp_gs2fs = extq.memory.forward_committor(basis_d, w, in_d, guess_fs, lag, mem)
        qm_fs2gs = extq.memory.backward_committor(basis_d, w, in_d, guess_fs, lag, mem)
        qm_gs2fs = extq.memory.backward_committor(basis_d, w, in_d, guess_gs, lag, mem)

        np.save(f"{base_dir}/dga_data/qp_fs2gs_lag1000_mem4.npy", qp_fs2gs)
        np.save(f"{base_dir}/dga_data/qp_gs2fs_lag1000_mem4.npy", qp_gs2fs)
        np.save(f"{base_dir}/dga_data/qm_fs2gs_lag1000_mem4.npy", qm_fs2gs)
        np.save(f"{base_dir}/dga_data/qm_gs2fs_lag1000_mem4.npy", qm_gs2fs)

        qp_fs2gs_withc = extq.memory.forward_committor(basis_d_withc, w, in_d_withc, guess_gs, lag, mem)
        qp_gs2fs_withc = extq.memory.forward_committor(basis_d_withc, w, in_d_withc, guess_fs, lag, mem)
        # qm boundary conditions should be 1 in A \cup \omega
        qm_fs2gs_withc = extq.memory.backward_committor(basis_d_withc, w, in_d_withc, guess_fs_withc, lag, mem)
        qm_gs2fs_withc = extq.memory.backward_committor(basis_d_withc, w, in_d_withc, guess_gs_withc, lag, mem)

        np.save(f"{base_dir}/dga_data/qp_fs2gs_lag1000_mem4_withc.npy", qp_fs2gs_withc)
        np.save(f"{base_dir}/dga_data/qp_gs2fs_lag1000_mem4_withc.npy", qp_gs2fs_withc)
        np.save(f"{base_dir}/dga_data/qm_fs2gs_lag1000_mem4_withc.npy", qm_fs2gs_withc)
        np.save(f"{base_dir}/dga_data/qm_gs2fs_lag1000_mem4_withc.npy", qm_gs2fs_withc)
    else:
        qp_fs2gs_withc = np.load(f"{base_dir}/dga_data/qp_fs2gs_lag1000_mem4_withc.npy")
        qp_gs2fs_withc = np.load(f"{base_dir}/dga_data/qp_gs2fs_lag1000_mem4_withc.npy")
        qm_fs2gs_withc = np.load(f"{base_dir}/dga_data/qm_fs2gs_lag1000_mem4_withc.npy")
        qm_gs2fs_withc = np.load(f"{base_dir}/dga_data/qm_gs2fs_lag1000_mem4_withc.npy")
        qp_fs2gs = np.load(f"{base_dir}/dga_data/qp_fs2gs_lag1000_mem4.npy")
        qp_gs2fs = np.load(f"{base_dir}/dga_data/qp_gs2fs_lag1000_mem4.npy")
        qm_fs2gs = np.load(f"{base_dir}/dga_data/qm_fs2gs_lag1000_mem4.npy")
        qm_gs2fs = np.load(f"{base_dir}/dga_data/qm_gs2fs_lag1000_mem4.npy")

    for q in [qp_fs2gs_withc, qp_gs2fs_withc, qm_fs2gs_withc, qm_fs2gs_withc, qp_fs2gs, qp_gs2fs, qm_fs2gs, qm_gs2fs]:
        q = [np.nan_to_num(traj, copy=False).clip(min=0, max=1) for traj in q]

    logging.info("Computing TPT rates...")
    rate_lags = np.array(
        [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000], dtype=int
    )
    rates_lags_fs2gs = tpt_rate(
        rate_lags, traj_inds, in_fs, in_gs, qp_fs2gs, qm_fs2gs, weights, in_d
    )
    rates_lags_gs2fs = tpt_rate(
        rate_lags, traj_inds, in_gs, in_fs, qp_gs2fs, qm_gs2fs, weights, in_d
    )
    # reaction coordinate needs to be 0 in A \cup \omega
    rates_lags_fs2gs_withc = tpt_rate(
        rate_lags, traj_inds, in_fs + in_c, in_gs, qp_fs2gs_withc, qm_fs2gs_withc, weights, in_d_withc
    )
    rates_lags_gs2fs_withc = tpt_rate(
        rate_lags, traj_inds, in_gs + in_c, in_fs, qp_gs2fs_withc, qm_gs2fs_withc, weights, in_d_withc
    )

    for i, (r, r_withc) in enumerate(zip(rates_lags_fs2gs, rates_lags_fs2gs_withc)):
        percent_through = r_withc / (r_withc + r)
        logging.info("Percent flux fs->gs through unfolded: %s", percent_through * 100)
        logging.info("Percent flux fs->gs not unfolded: %s", (1 - percent_through) * 100)
    for i, (r, r_withc) in enumerate(zip(rates_lags_gs2fs, rates_lags_gs2fs_withc)):
        percent_through = r_withc / (r_withc + r)
        logging.info("Percent flux gs->fs through unfolded: %s", percent_through * 100)
        logging.info("Percent flux gs->fs not unfolded: %s", (1 - percent_through) * 100)


    # np.save(f"{base_dir}/dga_data/rates_fs2gs.npy", rates_lags_fs2gs)
    # np.save(f"{base_dir}/dga_data/rates_gs2fs.npy", rates_lags_gs2fs)
    # np.save(f"{base_dir}/dga_data/rates_fs2gs_withc.npy", rates_lags_fs2gs_withc)
    # np.save(f"{base_dir}/dga_data/rates_gs2fs_withc.npy", rates_lags_gs2fs_withc)


if __name__ == "__main__":
    main()
