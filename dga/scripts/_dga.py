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


def forward_committor_coeffs(
    basis, weights, in_domain, guess, lag, test_basis=None
):
    """Solve the forward Feynman-Kac formula using DGA.

    Parameters
    ----------
    basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float
        Basis for estimating the solution of the Feynman-Kac formula.
        Must be zero outside of the domain.
    weights : list of (n_frames[i],) ndarray of float
        Change of measure to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    function : list of (n_frames[i]-1,) ndarray of float
        Function to integrate. Note that is defined over transitions,
        not frames.
    guess : list of (n_frames[i],) ndarray of float
        Guess of the solution. Must obey boundary conditions.
    lag : int
        DGA lag time in units of frames.
    test_basis : list of (n_frames[i], n_basis) ndarray or sparse matrix of float, optional
        Test basis against which to minimize the error. Must have the
        same dimension as the basis used to estimate the solution.
        If None, use the basis that is used to estimate the solution.

    Returns
    -------
    list of (n_frames[i],) ndarray
        Estimate of the solution of the forward Feynman-Kac formula at
        each frame of the trajectory.

    """
    function = np.zeros(len(weights))
    assert lag > 0
    if test_basis is None:
        test_basis = basis
    n_basis = None
    a = 0.0
    b = 0.0
    for x, y, w, d, f, g in zip(test_basis, basis, weights, in_domain, function, guess):
        n_frames = x.shape[0]
        n_basis = x.shape[1] if n_basis is None else n_basis
        f = np.broadcast_to(f, n_frames - 1)
        assert x.shape == (n_frames, n_basis)
        assert y.shape == (n_frames, n_basis)
        assert w.shape == (n_frames,)
        assert d.shape == (n_frames,)
        assert f.shape == (n_frames - 1,)
        assert g.shape == (n_frames,)
        assert np.all(w[max(0, n_frames - lag) :] == 0.0)
        if n_frames <= lag:
            continue
        iy = np.minimum(np.arange(lag, n_frames), extq.stop.forward_stop(d)[:-lag])
        intf = np.concatenate([np.zeros(1), np.cumsum(f)])
        integral = intf[iy] - intf[:-lag]
        wx = extq.linalg.scale_rows(w[:-lag], x[:-lag])
        a += wx.T @ (y[iy] - y[:-lag])
        b -= wx.T @ (g[iy] - g[:-lag] + integral)
    coeffs = extq.linalg.solve(a, b)
    return coeffs


def run_ivac(raw_feats):
    minlag = 10
    maxlag = 1000
    livac = ivac.LinearIVAC(minlag, maxlag, nevecs=10)
    livac.fit(raw_feats)
    ivac_trajs = livac.transform(raw_feats)
    with open(f"{base_dir}/dga_data/ivac.pkl", mode="wb+") as f:
        dill.dump(livac, f)
    np.save(f"{base_dir}/dga_data/ivac_10d_10-1000.npy", ivac_trajs)
    return ivac_trajs


def cluster(ivac_trajs):
    k = 300
    km = MiniBatchKMeans(k)
    km.fit(np.concatenate(ivac_trajs))
    centers = km.cluster_centers_
    neighbors = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    neighbors.fit(centers)
    with open(f"{base_dir}/dga_data/nneighbors.pkl", mode="wb+") as f:
        dill.dump(neighbors, f)

    dtraj = neighbors.kneighbors(np.concatenate(ivac_trajs), return_distance=False)
    np.save(f"{base_dir}/dga_data/dtraj_300.npy", dtraj)
    return dtraj.squeeze()


def make_basis(dtraj, in_fs, in_gs, traj_lens, traj_inds):
    basis_arr = make_sparse_basis(dtraj)[:, :-1]
    basis_d_arr = basis_arr.copy()
    basis_d_arr[in_fs] = 0
    basis_d_arr[in_gs] = 0

    basis, basis_d = [], []
    curr = 0
    for t_len in traj_lens:
        basis.append(basis_arr[curr : curr + t_len])
        basis_d.append(basis_d_arr[curr : curr + t_len])
        curr += t_len

    guess_fs = np.split(in_fs.astype(float), traj_inds)
    guess_gs = np.split(in_gs.astype(float), traj_inds)

    logging.info("Length of basis %s, basis shape %s", len(basis), basis[0].shape)
    return basis, basis_d, guess_fs, guess_gs


def lag_weights(weights, lag):
    result = []
    for w in weights:
        w[len(w) - lag :] = 0
        result.append(w)
    return result


def tpt_rate(lags, rate_lags, traj_inds, in_A, in_B, qps, qms, weights, in_d):
    in_Ac = np.split((~in_A).astype(float), traj_inds)
    in_B = np.split(in_B.astype(float), traj_inds)
    qm_rc = [(1 - qm) for qm in qms[-1]]
    n_lags = len(rate_lags)
    rates_lags = np.zeros((len(lags), 4, n_lags))
    rcs = [qps[2], qm_rc, in_B, in_Ac]

    for i, (qp, qm) in enumerate(zip(qps, qms)):
        for j, rc in enumerate(rcs):
            for k, lag in enumerate(rate_lags):
                logging.debug(
                    "Computing rates for committor lag %s, and RC %s rate lag %s...",
                    lags[i],
                    j,
                    lag,
                )
                com = lag_weights(weights[-1], lag)
                rates_lags[i, j, k] = extq.tpt.rate(qp, qm, com, in_d, rc, lag)
    return rates_lags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("temp")
    parser.add_argument("--ivac", action='store_true')
    parser.add_argument("--km", action='store_true')
    parser.add_argument("-w", "--weights", action='store_true')
    args = parser.parse_args()
    home_dir = "/project/dinner/scguo/kaiB"
    global base_dir
    base_dir = f"/project/dinner/scguo/kaiB/dga/{args.temp}"

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
    logging.info("Making basis...")
    basis, basis_d, guess_fs, guess_gs = make_basis(
        dtraj, in_fs, in_gs, traj_lens, traj_inds
    )

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

    logging.info("Computing committors...")
    in_domain = ~(in_fs | in_gs)
    in_d = np.split(in_domain, traj_inds)
    # qp_fs2gs, qm_fs2gs, qp_gs2fs, qm_gs2fs = [], [], [], []
    # fs2gs_coeffs, gs2fs_coeffs = [], []
    # for lag, w in zip(lags, weights):
    #     logging.info("Lag %s", lag)
    #     coef = forward_committor_coeffs(basis_d, w, in_d, guess_gs, lag)
    #     fs2gs_coeffs.append(coef)
    #     qp_fs2gs.append([x @ coef + g for (x, g) in zip(basis_d, guess_gs)])

    #     coef = forward_committor_coeffs(basis_d, w, in_d, guess_fs, lag)
    #     gs2fs_coeffs.append(coef)
    #     qp_gs2fs.append([x @ coef + g for (x, g) in zip(basis_d, guess_fs)])

    #     qm_fs2gs.append(extq.dga.backward_committor(basis_d, w, in_d, guess_fs, lag))
    #     qm_gs2fs.append(extq.dga.backward_committor(basis_d, w, in_d, guess_gs, lag))

    # np.save(f"{base_dir}/dga_data/qp_fs2gs.npy", qp_fs2gs)
    # np.save(f"{base_dir}/dga_data/qp_gs2fs.npy", qp_gs2fs)
    # np.save(f"{base_dir}/dga_data/qm_fs2gs.npy", qm_fs2gs)
    # np.save(f"{base_dir}/dga_data/qm_gs2fs.npy", qm_gs2fs)
    # np.save(f"{base_dir}/dga_data/qp_fs2gs_coeffs.npy", fs2gs_coeffs)
    # np.save(f"{base_dir}/dga_data/qp_gs2fs_coeffs.npy", gs2fs_coeffs)

    # logging.info("Computing TPT rates...")
    # rate_lags = np.array(
    #     [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000], dtype=int
    # )
    # rates_lags_fs2gs = tpt_rate(
    #     lags, rate_lags, traj_inds, in_fs, in_gs, qp_fs2gs, qm_fs2gs, weights, in_d
    # )
    # np.save(f"{base_dir}/dga_data/rates_fs2gs.npy", rates_lags_fs2gs)
    # rates_lags_gs2fs = tpt_rate(
    #     lags, rate_lags, traj_inds, in_gs, in_fs, qp_gs2fs, qm_gs2fs, weights, in_d
    # )
    # np.save(f"{base_dir}/dga_data/rates_gs2fs.npy", rates_lags_gs2fs)

    mem = 4
    qp_gs2fs, qp_fs2gs = {}, {}
    for lag, w in zip(lags[2:5], weights[2:5]):
        logging.info(f"Computing committors with lag %s and mem %s", lag, mem)
        gs2fs = extq.memory.forward_committor(basis_d, w, in_d, guess_fs, lag, mem)
        qp_gs2fs[(lag, mem)] = gs2fs
        fs2gs = extq.memory.forward_committor(basis_d, w, in_d, guess_gs, lag, mem)
        qp_fs2gs[(lag, mem)] = fs2gs
    np.save(f"{base_dir}/dga_data/qp_gs2fs_lag2000_mem4.npy", qp_gs2fs[(2000, 4)])
    np.save(f"{base_dir}/dga_data/qp_gs2fs_lag1000_mem4.npy", qp_gs2fs[(1000, 4)])
    np.save(f"{base_dir}/dga_data/qp_gs2fs_lag500_mem4.npy", qp_gs2fs[(500, 4)])
    np.save(f"{base_dir}/dga_data/qp_fs2gs_lag2000_mem4.npy", qp_fs2gs[(2000, 4)])
    np.save(f"{base_dir}/dga_data/qp_fs2gs_lag1000_mem4.npy", qp_fs2gs[(1000, 4)])
    np.save(f"{base_dir}/dga_data/qp_fs2gs_lag500_mem4.npy", qp_fs2gs[(500, 4)])



if __name__ == "__main__":
    main()
