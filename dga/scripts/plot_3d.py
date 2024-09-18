import sys
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import prettypyplot as pplt
import seaborn

import extq

upside_path = "/project/dinner/scguo/upside2/"
upside_utils_dir = os.path.expanduser(upside_path + "/py")
sys.path.insert(0, upside_utils_dir)

pplt.load_cmaps()
plt.rcParams[
    "text.latex.preamble"
] = r"\usepackage{siunitx}\sisetup{detect-all}\usepackage{helvet}\usepackage{sansmath}\sansmath"
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = "cm"
# mpl.rcParams["mathtext.fontset"] = "stixsans"

lag = 1000
names = ["blue", "green", "orange", "r_rmsd", "qp"]
labels = dict(
    blue=r"$\alpha3_{\mathrm{fs}}\longleftrightarrow \beta4_{\mathrm{gs}}$",
    green=r"$\beta4_{\mathrm{fs}}\longleftrightarrow\alpha3_{\mathrm{gs}}$",
    orange=r"$\beta3_{\mathrm{fs}}\longleftrightarrow\alpha2_{\mathrm{gs}}$",
    r_rmsd=r"core RMSD (nm)",
    qp=r"$q_{\mathrm{gs}\rightarrow\mathrm{fs}}$",
)
lims = dict(blue=(-1, 1.3), green=(-1.3, 1), orange=(-1.3, 1), r_rmsd=(0, 1.0), qp=(0, 1))



def load_cvs(base_dir, n_s, n_i):
    raw_feats, fs_qtots, f_rmsds, p_rmsds, r_rmsds = [], [], [], [], []
    for i in range(n_s):
        for j in range(n_i):
            for iso in ("cis", "trans"):
                idx = f"{i:02}_{j:02}_{iso}"
                head = f"{idx}_dga"
                if not os.path.exists(f"{base_dir}/{idx}/outputs/{head}_raw_feats.pkl"):
                    continue
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
    return fs_qtots, f_rmsds, p_rmsds, r_rmsds


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


def adjust_forward_committor(forward_q, in_domain, lag):
    result_qp = []
    for qp, d in zip(forward_q, in_domain):
        n = len(d)
        assert len(qp) == n
        tp = extq.stop.forward_stop(d)
        iy = np.minimum(np.arange(lag, n), tp[:-lag])
        result_qp.append(qp[iy])
    return result_qp


def delay_cv(cv, lag):
    return np.concatenate([traj[:-lag] for traj in cv])


def scatter_plot(temp):
    base_dir = f"/project/dinner/scguo/kaiB/dga/{temp}"
    fs_qtots, f_rmsds, p_rmsds, r_rmsds = load_cvs(base_dir, 7, 32)
    c_green = [traj[3, :] - traj[2, :] for traj in fs_qtots]
    c_blue = [traj[5, :] - traj[4, :] for traj in fs_qtots]
    c_orange = [traj[1, :] - traj[0, :] for traj in fs_qtots]

    c_green_arr = np.concatenate(c_green)
    c_blue_arr = np.concatenate(c_blue)
    c_orange_arr = np.concatenate(c_orange)
    r_rmsd_arr = np.asarray(r_rmsds).ravel()
    qp_gs2fs = np.load(f"{base_dir}/dga_data/qp_gs2fs_lag1000_mem4.npy", allow_pickle=True)
    qp_arr = np.concatenate(qp_gs2fs)

    N = len(qp_arr)
    skip = 200
    sc_ind = np.random.choice(N, N // skip)

    figure_dir = "/project/dinner/scguo/kaiB/dga"
    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.add_subplot(projection="3d")
    sc = ax.scatter(
        c_green_arr[sc_ind],
        c_blue_arr[sc_ind],
        c_orange_arr[sc_ind],
        "o",
        c=qp_arr[sc_ind],
        cmap="nightfall",
        alpha=0.1,
        s=10,
        rasterized=True,
        vmin=0,
        vmax=1
    )
    ax.set_xlabel(labels["green"])
    ax.set_ylabel(labels["blue"])
    ax.set_zlabel(labels["orange"])
    ax.grid(True)
    cb = plt.colorbar(sc, ax=ax, shrink=0.5, location='top', pad=0, label=labels["qp"])
    cb.solids.set(alpha=1)
    ax.view_init(elev=30, azim=-70)
    fig.savefig(f"{figure_dir}/figures/t{temp}_cvs_qp.png", dpi=500, bbox_inches='tight', pad_inches=0.5)

    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.add_subplot(projection="3d")
    sc = ax.scatter(
        c_green_arr[sc_ind],
        c_blue_arr[sc_ind],
        qp_arr[sc_ind],
        "o",
        c=qp_arr[sc_ind],
        cmap="nightfall",
        alpha=0.1,
        s=10,
        rasterized=True,
        vmin=0,
        vmax=1
    )
    ax.set_xlabel(labels["green"])
    ax.set_ylabel(labels["blue"])
    ax.set_zlabel(labels["qp"])
    ax.grid(True)
    cb = plt.colorbar(sc, ax=ax, shrink=0.5, location='top', pad=0, label=labels["qp"])
    cb.solids.set(alpha=1)
    ax.view_init(elev=30, azim=-70)
    fig.savefig(f"{figure_dir}/figures/t{temp}_cvs_qp_qp.png", dpi=500, bbox_inches='tight', pad_inches=0.5)

    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.add_subplot(projection="3d")
    sc = ax.scatter(
        c_green_arr[sc_ind],
        c_blue_arr[sc_ind],
        c_orange_arr[sc_ind],
        "o",
        c=r_rmsd_arr[sc_ind],
        cmap="rocket",
        alpha=0.1,
        s=10,
        rasterized=True,
        vmin=0,
        vmax=1
    )
    ax.set_xlabel(labels["green"])
    ax.set_ylabel(labels["blue"])
    ax.set_zlabel(labels["orange"])
    ax.grid(True)
    cb = plt.colorbar(sc, ax=ax, shrink=0.5, location='top', pad=0, label=labels["r_rmsd"])
    cb.solids.set(alpha=1)
    ax.view_init(elev=30, azim=-70)
    fig.savefig(f"{figure_dir}/figures/t{temp}_cvs_r_rmsd.png", dpi=500, bbox_inches='tight', pad_inches=0.5)


    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.add_subplot(projection="3d")
    sc = ax.scatter(
        c_green_arr[sc_ind],
        c_blue_arr[sc_ind],
        qp_arr[sc_ind],
        "o",
        c=r_rmsd_arr[sc_ind],
        cmap="rocket",
        alpha=0.1,
        s=10,
        rasterized=True,
        vmin=0,
        vmax=1
    )
    ax.set_xlabel(labels["green"])
    ax.set_ylabel(labels["blue"])
    ax.set_zlabel(labels["qp"])
    ax.grid(True)
    cb = plt.colorbar(sc, ax=ax, shrink=0.5, location='top', pad=0, label=labels["r_rmsd"])
    cb.solids.set(alpha=1)
    ax.view_init(elev=30, azim=-70)
    fig.savefig(f"{figure_dir}/figures/t{temp}_cvs_qp_r_rmsd.png", dpi=500, bbox_inches='tight', pad_inches=0.5)


def main():
    temps = [87, 89, 91]
    # temps = [91]
    for t in temps:
        scatter_plot(t)


if __name__ == "__main__":
    main()
