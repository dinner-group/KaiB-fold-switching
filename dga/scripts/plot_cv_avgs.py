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

plt.style.use("custom")
pplt.load_cmaps()
plt.rcParams[
    "text.latex.preamble"
] = r"\usepackage{siunitx}\sisetup{detect-all}\usepackage{helvet}\usepackage{sansmath}\sansmath"
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = "cm"
# mpl.rcParams["mathtext.fontset"] = "stixsans"

lag = 1000
names = ["blue", "green", "orange", "r_rmsd", "n_cis", "qp"]
labels = dict(
    blue=r"$\alpha3_{\mathrm{fs}}\longleftrightarrow \beta4_{\mathrm{gs}}$",
    green=r"$\beta4_{\mathrm{fs}}\longleftrightarrow\alpha3_{\mathrm{gs}}$",
    orange=r"$\beta3_{\mathrm{fs}}\longleftrightarrow\alpha2_{\mathrm{gs}}$",
    r_rmsd=r"core RMSD (nm)",
    n_cis=r"$n_{\mathrm{cis}}$",
    qp=r"$q_{\mathrm{gs}\rightarrow\mathrm{fs}}$",
)
lims = dict(blue=(-1, 1.3), green=(-1.3, 1), orange=(-1.3, 1), r_rmsd=(0, 1.0), qp=(0, 1), n_cis=(0, 2))



def load_cvs(base_dir, n_s, n_i):
    omegas, fs_qtots, f_rmsds, p_rmsds, r_rmsds = [], [], [], [], []
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
                for k in range(2):
                    if not os.path.exists(f"{base_dir}/{idx}/outputs/{idx}_{k:02}_Omega.npy"):
                        continue
                    omegas.append(np.load(f"{base_dir}/{idx}/outputs/{idx}_{k:02}_Omega.npy"))
    return fs_qtots, f_rmsds, p_rmsds, r_rmsds, omegas


def delay_cv(cv, lag):
    return np.concatenate([traj[:-lag] for traj in cv])


def rev_delay_cv(cv, lag):
    return np.concatenate([traj[lag:] for traj in cv])


def plot_cvs(temp):
    home_dir = "/project/dinner/scguo/kaiB/dga"
    base_dir = f"{home_dir}/{temp}"
    fs_qtots, f_rmsds, p_rmsds, r_rmsds, omegas = load_cvs(base_dir, 7, 32)
    n_cis = []
    for omega in omegas:
        n_cis.append(np.sum((np.abs(omega[:, 3:]) <= (np.pi / 2.0)).astype(int), axis=1))
    c_green = [traj[3, :] - traj[2, :] for traj in fs_qtots]
    c_blue = [traj[5, :] - traj[4, :] for traj in fs_qtots]
    c_orange = [traj[1, :] - traj[0, :] for traj in fs_qtots]

    weights = np.load(f"{base_dir}/dga_data/weights.npy")[4]

    c_green_delay = rev_delay_cv(c_green, lag)
    c_blue_delay = rev_delay_cv(c_blue, lag)
    w_delay = delay_cv(weights, lag)
    # plot averages of different cvs
    fig, axes = plt.subplots(
        nrows=1, ncols=3, figsize=(6, 2.5), dpi=500, sharex=True, sharey=True, constrained_layout=True
    )

    xe = np.linspace(*lims["green"], 51)
    ye = np.linspace(*lims["blue"], 51)
    xc = (xe[1:] + xe[:-1]) / 2
    yc = (ye[1:] + ye[:-1]) / 2

    names = ["orange", "r_rmsd", "n_cis"]
    cvs = [c_orange, r_rmsds, n_cis]
    for ax, name, cv in zip(axes, names, cvs):
        func_delay = rev_delay_cv(cv, lag)
        hist = extq.projection.average2d(c_green_delay, c_blue_delay, func_delay, w_delay, xe, ye)
        vmin, vmax = lims[name]
        pc = ax.pcolormesh(xc, yc, hist.T, cmap="rocket", vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_ylabel(r"$\alpha3_{\mathrm{fs}}\longleftrightarrow \beta4_{\mathrm{gs}}$")
        ax.set_xlabel(r"$\beta4_{\mathrm{fs}}\longleftrightarrow\alpha3_{\mathrm{gs}}$")
        ax.label_outer()
        cb = plt.colorbar(pc, ax=ax, label=labels[name], location="top")
    fig.savefig(f"{home_dir}/figures/t{temp}_av_ot_rrmsd_ncis.pdf", bbox_inches="tight")


def main():
    temps = [87, 89, 91]
    for t in temps:
        plot_cvs(t)


if __name__ == "__main__":
    main()
