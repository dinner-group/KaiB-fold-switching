import glob
import numpy as np

home_dir = "/project/dinner/scguo/kaiB"


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


def get_trajfiles(temp):
    base_dir = f"{home_dir}/dga/new_{temp}"
    traj_files = []
    for i in range(1, 12289):
        j = i // 1000
        for iso in ("cis", "trans"):
            head = f"{base_dir}/{j:02}/{i:05}/outputs/{i:05}_{iso}"
            listed = glob.glob(f"{base_dir}/{j:02}/{i:05}/{iso}/*.h5")
            if len(listed) != 0:
                traj_files.extend(listed)
    return traj_files


def get_weights(weights, N, k, ids):
    sampled_w = np.concatenate(weights[:, -N::k])
    # set weight for cis/trans to be the same
    sampled_w = np.repeat(sampled_w, 2)
    return sampled_w[ids]


def load_cv(basename):
    arrays = []
    for file in sorted(glob.glob(f"{basename}_??.npy")):
        arrays.append(np.load(file))
    return np.concatenate(arrays)


def load_all(temp):
    base_dir = f"{home_dir}/dga/new_{temp}"
    data_dir = f"{base_dir}/data"
    weights = np.load(f"{data_dir}/mbar_weights.npy")
    # ids = get_traj_ids(base_dir)
    ids = np.loadtxt(f"{data_dir}/traj_ids.txt", dtype=int)
    N = 32000
    k = 1000
    sample_w = get_weights(weights, N, k, ids)

    fs_qtots = load_cv(f"{data_dir}/fs_qtots")
    q_gs_all = load_cv(f"{data_dir}/q_gs_all")
    q_gs = load_cv(f"{data_dir}/q_gs")
    q_fs_all = load_cv(f"{data_dir}/q_fs_all")
    q_fs = load_cv(f"{data_dir}/q_fs")
    q_core = load_cv(f"{data_dir}/q_core")
    omegas = load_cv(f"{data_dir}/omegas")
    pots = load_cv(f"{data_dir}/pots")
    f_rmsds = load_cv(f"{data_dir}/f_rmsds")
    p_rmsds = load_cv(f"{data_dir}/p_rmsds")

    c_green = fs_qtots[:, 4] - fs_qtots[:, 5]
    c_blue = fs_qtots[:, 6] - fs_qtots[:, 7]
    c_orange = fs_qtots[:, 2] - fs_qtots[:, 3]
    c_0 = fs_qtots[:, 0] - fs_qtots[:, 1]
    n_cis = np.sum((np.abs(omegas[..., 3:]) <= (np.pi / 2.0)).astype(int), axis=-1)
    p63_cis = (np.abs(omegas[..., 3]) <= (np.pi / 2.0)).astype(int)
    p70_cis = (np.abs(omegas[..., 4]) <= (np.pi / 2.0)).astype(int)
    p71_cis = (np.abs(omegas[..., 5]) <= (np.pi / 2.0)).astype(int)
    p72_cis = (np.abs(omegas[..., 6]) <= (np.pi / 2.0)).astype(int)

    cvs = dict(
        blue=c_blue,
        green=c_green,
        orange=c_orange,
        c0=c_0,
        q_gs_all=q_gs_all,
        q_fs_all=q_fs_all,
        q_gs=q_gs,
        q_fs=q_fs,
        q_core=q_core,
        n_cis=n_cis,
        pot=pots,
        f_rmsds=f_rmsds,
        p_rmsds=p_rmsds,
        q_diff=q_fs - q_gs,
        p63=p63_cis,
        p70=p70_cis,
        p71=p71_cis,
        p72=p72_cis,
    )
    return cvs, sample_w
