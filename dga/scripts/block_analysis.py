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
import ivac

upside_path = "/project/dinner/scguo/upside2"
upside_utils_dir = os.path.expanduser(upside_path + "/py")
sys.path.insert(0, upside_utils_dir)

base_dir = ""


def load_cvs(base_dir, n_s, n_i, block=None):
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

    if block is not None:
        raw_feats = _make_blocks(block, raw_feats)
        # fs_qtots is n_trajs, n_types, n_frames so need to transpose
        fs_qtots = [traj.T for traj in fs_qtots]
        fs_qtots = _make_blocks(block, fs_qtots)
        # and then undo
        fs_qtots = [traj.T for traj in fs_qtots]
        f_rmsds = _make_blocks(block, f_rmsds)
        p_rmsds = _make_blocks(block, p_rmsds)
        r_rmsds = _make_blocks(block, r_rmsds)

    return raw_feats, fs_qtots, f_rmsds, p_rmsds, r_rmsds


def _make_blocks(segment, trajs):
    traj_len = len(trajs[0])
    if segment == 1:
        return np.array(trajs)[:, : traj_len // 2]
    elif segment == 2:
        return np.array(trajs)[:, traj_len // 2 :]


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


def run_ivac(raw_feats):
    minlag = 10
    maxlag = 1000
    livac = ivac.LinearIVAC(minlag, maxlag, nevecs=10)
    livac.fit(raw_feats)
    ivac_trajs = livac.transform(raw_feats)
    np.save(f"{base_dir}/dga_data/ivac_10d_10-1000_block{block}.npy", ivac_trajs)
    return ivac_trajs


def cluster(ivac_trajs):
    k = 300
    km = MiniBatchKMeans(k)
    km.fit(np.concatenate(ivac_trajs))
    centers = km.cluster_centers_
    neighbors = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    neighbors.fit(centers)
    dtraj = neighbors.kneighbors(np.concatenate(ivac_trajs), return_distance=False)
    np.save(f"{base_dir}/dga_data/dtraj_300_block{block}.npy", dtraj)
    return dtraj.squeeze()


def _basis_helper(dtraj, in_dc):
    # basis_arr = make_sparse_basis(dtraj)[:, :-1]
    basis_arr = make_dense_basis(dtraj)[:, :-1]
    basis_arr[in_dc] = 0
    return scipy.sparse.csr_matrix(basis_arr)


def make_basis(dtraj, in_fs, in_gs, traj_lens, traj_inds):
    basis_arr = make_sparse_basis(dtraj)[:, :-1]
    # basis_d_arr = _basis_helper(dtraj, (in_fs | in_gs))
    basis_d_arr = basis_arr.copy()
    basis_d_arr[in_fs] = 0
    basis_d_arr[in_gs] = 0

    basis, basis_d, = [], []
    curr = 0
    for t_len in traj_lens:
        basis.append(basis_arr[curr : curr + t_len])
        basis_d.append(basis_d_arr[curr : curr + t_len])
        curr += t_len

    logging.info("Length of basis %s, basis shape %s", len(basis_d), basis_d[0].shape)
    return basis, basis_d


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
    parser.add_argument("--ivac", action="store_true")
    parser.add_argument("--km", action="store_true")
    parser.add_argument("-w", "--weights", action="store_true")
    parser.add_argument("--block", type=int)
    args = parser.parse_args()
    home_dir = "/project/dinner/scguo/kaiB"
    global base_dir
    base_dir = f"/project/dinner/scguo/kaiB/dga/{args.temp}"
    global block
    block = args.block

    ivac_precomputed = args.ivac
    km_precomputed = args.km
    weights_precomputed = args.weights

    logging.basicConfig(
        filename=f"dga_{args.temp}.log",
        encoding="utf-8",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )

    logging.info("Loading features and CVs")
    # take either first half or second half of trajectory
    if args.block != 1 and args.block != 2:
        raise ValueError
    raw_feats, fs_qtots, f_rmsds, p_rmsds, r_rmsds = load_cvs(base_dir, 7, 32, block=block)
    logging.info("Finished loading.")
    logging.info("Len(raw_feats) %s, raw_feats.shape %s", len(raw_feats), raw_feats[0].shape)
    logging.info("Len(fs_qtots) %s, fs_qtots.shape %s", len(fs_qtots), fs_qtots[0].shape)

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
    in_c = r_rmsd_arr > 0.7

    if not ivac_precomputed:
        logging.info("Running IVAC...")
        ivac_trajs = run_ivac(raw_feats)
    else:
        logging.info(
            "Loading IVAC data from %s", f"{base_dir}/dga_data/ivac_10d_10-1000_block{block}.npy"
        )
        ivac_trajs = np.load(f"{base_dir}/dga_data/ivac_10d_10-1000_block{block}.npy")
    if not km_precomputed:
        logging.info("Clustering...")
        dtraj = cluster(ivac_trajs)
    else:
        logging.info(
            "Loading clustered data from %s", f"{base_dir}/dga_data/dtraj_300_block{block}.npy"
        )
        dtraj = np.load(f"{base_dir}/dga_data/dtraj_300_block{block}.npy").squeeze()

    traj_lens, traj_inds = split_indices(raw_feats)

    lags = np.array([100, 200, 500, 1000, 2000, 5000], dtype=int)

    guess_fs = np.split(in_fs.astype(float), traj_inds)
    guess_gs = np.split(in_gs.astype(float), traj_inds)
    in_domain = ~(in_fs | in_gs)
    in_d = np.split(in_domain, traj_inds)

    logging.info("Making basis...")

    basis, basis_d = make_basis(
        dtraj, in_fs, in_gs, traj_lens, traj_inds
    )
    if not weights_precomputed:
        logging.info("Computing weights...")
        weights = []
        for lag in lags:
            weights.append(extq.dga.reweight(basis, lag))
        logging.info(
            "Saving weights to %s", f"{base_dir}/dga_data/weights_block{block}.npy"
        )
        np.save(
            f"{base_dir}/dga_data/weights_block{block}.npy", weights, allow_pickle=True
        )
    else:
        logging.info(
            "Loading precomputed weights from %s", f"{base_dir}/dga_data/weights_block{block}.npy"
        )
        weights = np.load(f"{base_dir}/dga_data/weights_block{block}.npy", allow_pickle=True)
    mem = 4
    qp_gs2fs, qp_fs2gs = {}, {}
    for lag, w in zip(lags[2:5], weights[2:5]):
        logging.info(f"Computing committors with lag %s and mem %s", lag, mem)
        gs2fs = extq.memory.forward_committor(basis_d, w, in_d, guess_fs, lag, mem)
        qp_gs2fs[(lag, mem)] = gs2fs
        fs2gs = extq.memory.forward_committor(basis_d, w, in_d, guess_gs, lag, mem)
        qp_fs2gs[(lag, mem)] = fs2gs
    np.save(f"{base_dir}/dga_data/qp_gs2fs_lag2000_mem4_block{block}.npy", qp_gs2fs[(2000, 4)])
    np.save(f"{base_dir}/dga_data/qp_gs2fs_lag1000_mem4_block{block}.npy", qp_gs2fs[(1000, 4)])
    np.save(f"{base_dir}/dga_data/qp_gs2fs_lag500_mem4_block{block}.npy", qp_gs2fs[(500, 4)])
    np.save(f"{base_dir}/dga_data/qp_fs2gs_lag2000_mem4_block{block}.npy", qp_fs2gs[(2000, 4)])
    np.save(f"{base_dir}/dga_data/qp_fs2gs_lag1000_mem4_block{block}.npy", qp_fs2gs[(1000, 4)])
    np.save(f"{base_dir}/dga_data/qp_fs2gs_lag500_mem4_block{block}.npy", qp_fs2gs[(500, 4)])


if __name__ == "__main__":
    main()
