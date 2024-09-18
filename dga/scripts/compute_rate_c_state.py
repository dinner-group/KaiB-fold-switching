import os
import sys
import dill
import glob
import logging
import argparse
from joblib import Memory

import numpy as np
import scipy

import extq
from dga import make_sparse_basis, compute_ivac, kmeans_cluster

upside_path = "/project/dinner/scguo/upside2"
upside_utils_dir = os.path.expanduser(upside_path + "/py")
sys.path.insert(0, upside_utils_dir)

# hyperparameters
qlag, qmem = 100, 1
k = 500
minlag, maxlag, nevecs = 1, 200, 6

parser = argparse.ArgumentParser()
parser.add_argument("temp")
args = parser.parse_args()
home_dir = "/project/dinner/scguo/kaiB"
base_dir = f"{home_dir}/dga/new_{args.temp}"
data_dir = f"{base_dir}/data"
memory = Memory(data_dir, verbose=0)


def load_cv(basename):
    arrays = []
    for file in sorted(glob.glob(f"{basename}_??.npy")):
        arrays.append(np.load(file))
    return np.concatenate(arrays)


def get_traj_ids(base_dir):
    ids = []
    k = 0
    for i in range(1, 12289):
        j = i // 1000
        for iso in ("cis", "trans"):
            head = f"{base_dir}/{j:02}/{i:05}/outputs/{i:05}_{iso}"
            if len(glob.glob(f"{base_dir}/{j:02}/{i:05}/{iso}/*.up")) > 0:
                ids.append(k)
            k += 1
    return np.array(ids)


def get_weights(weights, N, k, ids):
    sampled_w = np.concatenate(weights[:, -N::k])
    # set weight for cis/trans to be the same
    sampled_w = np.repeat(sampled_w, 2)
    return sampled_w[ids]


@memory.cache
def make_basis(dtraj, in_d):
    basis_d_arr = make_sparse_basis(dtraj)
    basis_d_arr = scipy.sparse.csr_array(basis_d_arr.multiply(in_d.ravel()[..., None]))
    # remove basis functions which are 0 everywhere
    mask = np.ravel(np.sum(basis_d_arr != 0, axis=0).astype(bool))
    basis_d_arr = basis_d_arr[:, mask]
    basis_d = []
    n_traj, n_frame = in_d.shape
    for i in range(n_traj):
        basis_d.append(basis_d_arr[i * n_frame : (i + 1) * n_frame])
    return basis_d


@memory.cache
def compute_committor(basis_d, sample_w, lag, mem, in_d, guess_qp, guess_qm):
    w = np.array(np.broadcast_to(sample_w[..., None], guess_qp.shape))
    w[:, -lag:] = 0
    qp = extq.memory.forward_committor(basis_d, w, in_d, guess_qp, lag, mem)
    qm = extq.memory.backward_committor(basis_d, w, in_d, guess_qm, lag, mem)
    return qp, qm


@memory.cache
def tpt_rate(sample_w, rate_lags, in_A, in_B, qp, qm, in_d):
    qm_rc = [(1 - q) for q in qm]
    n_lags = len(rate_lags)
    rates_lags = np.zeros((4, n_lags))
    rcs = [qp, qm_rc, in_B, 1 - in_A]

    for j, rc in enumerate(rcs):
        for k, lag in enumerate(rate_lags):
            w = np.array(np.broadcast_to(sample_w[..., None], in_A.shape))
            w[:, -lag:] = 0
            rates_lags[j, k] = extq.tpt.rate(qp, qm, w, in_d, rc, lag)
    return rates_lags


def main():
    logging.basicConfig(
        filename=f"dga_{args.temp}.log",
        encoding="utf-8",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )

    logging.info("Loading features and CVs")
    fs_qtots = load_cv(f"{data_dir}/fs_qtots")
    f_rmsds = load_cv(f"{data_dir}/f_rmsds")
    r_rmsds = load_cv(f"{data_dir}/r_rmsds")
    p_rmsds = load_cv(f"{data_dir}/p_rmsds")
    q_gs_all = load_cv(f"{data_dir}/q_gs_all")
    q_gs = load_cv(f"{data_dir}/q_gs")
    q_fs_all = load_cv(f"{data_dir}/q_fs_all")
    q_fs = load_cv(f"{data_dir}/q_fs")
    q_core = load_cv(f"{data_dir}/q_core")
    omegas = load_cv(f"{data_dir}/omegas")
    weights = np.load(f"{data_dir}/mbar_weights.npy")
    raw_contacts = load_cv(f"{data_dir}/raw_contacts")

    ids = get_traj_ids(base_dir)
    logging.info("Number of trajectories: %s", len(ids))
    N = 32000
    skip = 1000
    sample_w = get_weights(weights, N, skip, ids)
    logging.info("Finished loading.")

    c_green = fs_qtots[:, 5] - fs_qtots[:, 4]
    c_blue = fs_qtots[:, 7] - fs_qtots[:, 6]
    c_orange = fs_qtots[:, 3] - fs_qtots[:, 2]
    c_0 = fs_qtots[:, 1] - fs_qtots[:, 0]

    in_fs = (
        (c_green < -0.65)
        & (c_blue < -0.8)
        & (c_orange < -0.85)
        & (q_fs_all > 0.61)
        & (q_fs > 0.75)
        & (f_rmsds < 0.35)
        & (q_core > 0.65)
    )
    in_gs = (
        (c_green > 0.55)
        & (c_blue > 0.90)
        & (c_orange > 0.75)
        & (q_gs_all > 0.62)
        & (q_gs > 0.65)
        & (p_rmsds < 0.45)
        & (q_core > 0.7)
        & (c_0 > 0.5)
    )
    in_c = (
        (q_core < 0.16)
        & (np.abs(c_green) < 0.5)
        & (np.abs(c_blue) < 0.5)
        & (np.abs(c_orange) < 0.5)
    )

    logging.info(
        "Running IVAC with minlag %s maxlag %s and nevecs %s",
        minlag,
        maxlag,
        nevecs - 1,
    )
    _, ivac_trajs = compute_ivac(raw_contacts, minlag, maxlag, nevecs, adjust=True)

    omega_features = np.cos(omegas[..., 3:])
    feature_trajs = np.concatenate(
        [np.asarray(ivac_trajs)[..., 1:], omega_features], axis=-1
    )
    logging.info("Clustering with k = %s", k)
    dtraj = kmeans_cluster(feature_trajs, k)

    logging.info("Loading committors...")
    with open(f"{data_dir}/qp_gs2fs.pkl", mode="rb") as f:
        qp_gs2fs = dill.load(f)[(qlag, qmem)]
    with open(f"{data_dir}/qm_gs2fs.pkl", mode="rb") as f:
        qm_gs2fs = dill.load(f)[(qlag, qmem)]

    in_d = ~(in_fs | in_gs)
    in_d_withc = ~(in_fs | in_gs | in_c)
    guess_fs = in_fs.astype(float)
    guess_gs_withc = (in_gs + in_c).astype(float)
    guess_fs_withc = (in_fs + in_c).astype(float)
    guess_gs = (in_gs).astype(float)

    logging.info("Making basis...")
    basis_d = make_basis(dtraj, in_d)
    basis_d_withc = make_basis(dtraj, in_d_withc)

    logging.info("Computing committors...")
    # qm boundary conditions should be 1 in A \cup \omega
    qp_gs2fs_withc, qm_gs2fs_withc = compute_committor(
        basis_d_withc, sample_w, qlag, qmem, in_d_withc, guess_fs, guess_gs_withc
    )
    qp_fs2gs, qm_fs2gs = compute_committor(
        basis_d, sample_w, qlag, qmem, in_d, guess_gs, guess_fs
    )
    qp_fs2gs_withc, qm_fs2gs_withc = compute_committor(
        basis_d_withc, sample_w, qlag, qmem, in_d_withc, guess_gs, guess_fs_withc
    )
    np.save(f"{data_dir}/qp_fs2gs.npy", qp_fs2gs)
    np.save(f"{data_dir}/qm_fs2gs.npy", qm_fs2gs)
    np.save(f"{data_dir}/qp_fs2gs_withc.npy", qp_fs2gs_withc)
    np.save(f"{data_dir}/qm_fs2gs_withc.npy", qm_fs2gs_withc)
    np.save(f"{data_dir}/qp_gs2fs_withc.npy", qp_gs2fs_withc)
    np.save(f"{data_dir}/qm_gs2fs_withc.npy", qm_gs2fs_withc)


    for q in [
        qp_gs2fs_withc,
        qm_gs2fs_withc,
        qp_gs2fs,
        qm_gs2fs,
        qp_fs2gs_withc,
        qm_fs2gs_withc,
        qp_fs2gs,
        qm_fs2gs,
    ]:
        q = [np.nan_to_num(traj, copy=False).clip(min=0, max=1) for traj in q]

    logging.info("Computing TPT rates...")
    rate_lags = np.array([5, 10, 20, 50, 100, 200], dtype=int)

    rates_lags_gs2fs = tpt_rate(
        sample_w,
        rate_lags,
        in_gs.astype(float),
        in_fs.astype(float),
        qp_gs2fs,
        qm_gs2fs,
        in_d,
    )
    rates_lags_fs2gs = tpt_rate(
        sample_w,
        rate_lags,
        in_fs.astype(float),
        in_gs.astype(float),
        qp_fs2gs,
        qm_fs2gs,
        in_d,
    )
    # reaction coordinate needs to be 0 in A \cup \omega
    rates_lags_gs2fs_withc = tpt_rate(
        sample_w,
        rate_lags,
        (in_gs + in_c).astype(float),
        in_fs.astype(float),
        qp_gs2fs_withc,
        qm_gs2fs_withc,
        in_d_withc,
    )
    rates_lags_fs2gs_withc = tpt_rate(
        sample_w,
        rate_lags,
        (in_fs + in_c).astype(float),
        in_gs.astype(float),
        qp_fs2gs_withc,
        qm_fs2gs_withc,
        in_d_withc,
    )

    for i, (r, r_withc) in enumerate(zip(rates_lags_gs2fs, rates_lags_gs2fs_withc)):
        percent_through = r_withc / (r_withc + r)
        logging.info("Percent flux gs->fs through unfolded: %s", percent_through * 100)
        logging.info(
            "Percent flux gs->fs not unfolded: %s", (1 - percent_through) * 100
        )
    for i, (r, r_withc) in enumerate(zip(rates_lags_fs2gs, rates_lags_fs2gs_withc)):
        percent_through = r_withc / (r_withc + r)
        logging.info("Percent flux fs->gs through unfolded: %s", percent_through * 100)
        logging.info(
            "Percent flux fs->gs not unfolded: %s", (1 - percent_through) * 100
        )

    # np.save(f"{data_dir}/rates_gs2fs.npy", rates_lags_gs2fs)
    # np.save(f"{data_dir}/rates_gs2fs_withc.npy", rates_lags_gs2fs_withc)
    # np.save(f"{data_dir}/rates_fs2gs.npy", rates_lags_fs2gs)
    # np.save(f"{data_dir}/rates_fs2gs_withc.npy", rates_lags_fs2gs_withc)


if __name__ == "__main__":
    main()
