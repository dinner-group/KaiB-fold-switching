import os
import sys
import dill
import glob
import logging
import argparse
from collections import OrderedDict
from joblib import Memory

import numpy as np
import scipy

import extq

upside_path = "/project/dinner/scguo/upside2"
upside_utils_dir = os.path.expanduser(upside_path + "/py")
sys.path.insert(0, upside_utils_dir)

# hyperparameters
lags = np.array([1, 2, 5, 10], dtype=int)
qlag, qmem = 100, 1

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


def compute_current(lag, cv, w):
    w[..., -lag:] = 0
    j = extq.tpt.current(qp_gs2fs, qm_gs2fs, w, in_d, cvs[cv], lag)
    return j


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

logging.info("Loading committors...")
in_d = ~(in_fs | in_gs | in_c)

# with open(f"{data_dir}/qp_gs2fs.pkl", mode="rb") as f:
#     qp_gs2fs = dill.load(f)[(qlag, qmem)]
#     qp_gs2fs = [np.nan_to_num(traj) for traj in qp_gs2fs]
# with open(f"{data_dir}/qm_gs2fs.pkl", mode="rb") as f:
#     qm_gs2fs = dill.load(f)[(qlag, qmem)]
#     qm_gs2fs = [np.nan_to_num(traj) for traj in qm_gs2fs]
qp_gs2fs = np.load(f"{data_dir}/qp_gs2fs_withc.npy")
qm_gs2fs = np.load(f"{data_dir}/qm_gs2fs_withc.npy")
qp_gs2fs = [np.nan_to_num(traj) for traj in qp_gs2fs]
qm_gs2fs = [np.nan_to_num(traj) for traj in qm_gs2fs]

cvs = OrderedDict(dict(
    blue=c_blue,
    green=c_green,
    orange=c_orange,
    q_gs_all=q_gs_all,
    q_gs=q_gs,
    q_fs_all=q_fs_all,
    q_fs=q_fs,
    q_core=q_core,
    qp=qp_gs2fs,
))
names = cvs.keys()

logging.info("Computing currents...")
for name in names:
    logging.info("CV %s", name)
    j_lags = []
    for lag in lags:
        w = np.array(np.broadcast_to(sample_w[..., None], q_gs.shape))
        # current = compute_current(lag, name, w)
        w[..., -lag:] = 0
        current = extq.tpt.current(qp_gs2fs, qm_gs2fs, w, in_d, cvs[name], lag)
        j_lags.append(current)
    np.save(f"{data_dir}/j_gs2fs_withc_{name}.npy", j_lags)
