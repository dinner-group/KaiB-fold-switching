import os
import glob
import sys
import dill
import logging
import argparse
from joblib import Memory

import numpy as np
import scipy
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

import extq
import ivac

upside_path = "/project/dinner/scguo/upside2"
upside_utils_dir = os.path.expanduser(upside_path + "/py")
sys.path.insert(0, upside_utils_dir)

# hyperparameters
k = 500
minlag, maxlag, nevecs = 1, 200, 6
lags = np.array([50, 100, 200], dtype=int)
mems = np.array([1, 4, 9], dtype=int)

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


@memory.cache
def compute_ivac(trajs, minlag, maxlag, nevecs, **kwargs):
    ivac_obj = ivac.LinearIVAC(minlag, maxlag, nevecs=nevecs, **kwargs)
    ivac_obj.fit(trajs)
    ivac_trajs = ivac_obj.transform(trajs)
    return ivac_obj, ivac_trajs


@memory.cache
def kmeans_cluster(trajs, k, batch_size=2048):
    flattened_trajs = np.concatenate(trajs)
    kmeans_obj = MiniBatchKMeans(n_clusters=k, batch_size=batch_size)
    kmeans_obj.fit(flattened_trajs)
    nneighbors = NearestNeighbors(n_neighbors=1)
    nneighbors.fit(kmeans_obj.cluster_centers_)
    dtrajs = nneighbors.kneighbors(flattened_trajs, 1, return_distance=False)
    return dtrajs.squeeze()


@memory.cache
def make_basis(dtraj, in_fs, in_gs):
    basis_d_arr = make_sparse_basis(dtraj)
    basis_d_arr[in_fs.ravel()] = 0
    basis_d_arr[in_gs.ravel()] = 0
    # remove basis functions which are 0 everywhere
    mask = np.ravel(np.sum(basis_d_arr != 0, axis=0).astype(bool))
    basis_d_arr = basis_d_arr[:, mask]
    basis_d = []
    n_traj, n_frame = in_fs.shape
    for i in range(n_traj):
        basis_d.append(basis_d_arr[i * n_frame : (i + 1) * n_frame])
    return basis_d


@memory.cache
def compute_committor(lag, mem):
    qp = extq.memory.forward_committor(basis_d, w, in_d, in_fs.astype(float), lag, mem)
    qm = extq.memory.backward_committor(basis_d, w, in_d, in_gs.astype(float), lag, mem)
    return qp, qm

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

    logging.info("Running IVAC with minlag %s maxlag %s and nevecs %s", minlag, maxlag, nevecs - 1)
    ivac_obj, ivac_trajs = compute_ivac(raw_contacts, minlag, maxlag, nevecs, adjust=True)
    logging.info("First 10 implied timescales: %s", ivac_obj.its[:10])


    omega_features = np.cos(omegas[..., 3:])
    feature_trajs = np.concatenate([np.asarray(ivac_trajs)[..., 1:], omega_features], axis=-1)
    logging.info("Clustering with k = %s", k)
    dtraj = kmeans_cluster(feature_trajs, k)

    logging.info("Making basis...")
    basis_d = make_basis(dtraj, in_fs, in_gs)

    in_d = ~(in_gs | in_fs)
    logging.info("Computing committors...")

    qp_gs2fs, qm_gs2fs = {}, {}
    for lag in lags:
        w = np.array(np.broadcast_to(sample_w[..., None], q_gs.shape))
        w[:, -lag:] = 0
        for mem in mems:
            logging.info(f"Computing committors with lag %s and mem %s", lag, mem)
            qp, qm = compute_committor(lag, mem)
            qp_gs2fs[(lag, mem)] = qp
            qm_gs2fs[(lag, mem)] = qm
    # np.save( qp_gs2fs, allow_pickle=True)
    # np.save(f"{data_dir}/qm_gs2fs.pkl", qm_gs2fs, allow_pickle=True)
    with open(f"{data_dir}/qp_gs2fs.pkl", mode='wb') as f:
        dill.dump(qp_gs2fs, f)
    with open(f"{data_dir}/qm_gs2fs.pkl", mode='wb') as f:
        dill.dump(qm_gs2fs, f)


if __name__ == "__main__":
    main()
