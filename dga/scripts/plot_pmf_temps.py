import sys
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import prettypyplot as pplt

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
mpl.rcParams["mathtext.fontset"] = "stixsans"

lag = 5000


def load_cvs(base_dir, n_s, n_i):
    fs_qtots = []
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
    return fs_qtots


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


def compute_hist(temp):
    base_dir = f"/project/dinner/scguo/kaiB/dga/{temp}"
    fs_qtots = load_cvs(base_dir, 7, 32)
    c_green = [traj[3, :] - traj[2, :] for traj in fs_qtots]
    c_blue = [traj[5, :] - traj[4, :] for traj in fs_qtots]

    weights = np.load(f"{base_dir}/dga_data/weights.npy", allow_pickle=True)

    w_delay = [t[:-lag] for t in weights[-1]]
    cv1_delay = [t[lag:] for t in c_green]
    cv2_delay = [t[lag:] for t in c_blue]

    xe = np.linspace(-1.3, 1, 101)
    ye = np.linspace(-1, 1.3, 101)
    hist = extq.projection.density2d(
        cv1_delay, cv2_delay, w_delay, xe, ye
    )

    return hist


def plot_pmf(hists):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(7, 2.25),
        dpi=500,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    xe = np.linspace(-1.3, 1, 101)
    ye = np.linspace(-1, 1.3, 101)
    xc = (xe[1:] + xe[:-1]) / 2
    yc = (ye[1:] + ye[:-1]) / 2
    clines = np.arange(0, 13, 2)

    # fs to gs committors
    for ax, hist in zip(axes, hists):
        offset = np.min(-np.log(hist))
        pmf = -np.log(hist) - offset
        pc = ax.pcolormesh(
            xc, yc, pmf.T, cmap="iridescent_r", vmin=0, vmax=13, rasterized=True
        )
        ax.contour(xc, yc, pmf.T, colors="grey", levels=clines, linewidths=1)
        ax.set_ylabel(
            r"$\alpha3_{\mathrm{fs}}\longleftrightarrow \beta4_{\mathrm{gs}}$"
        )
        ax.set_xlabel(r"$\beta4_{\mathrm{fs}}\longleftrightarrow\alpha3_{\mathrm{gs}}$")
        ax.label_outer()
    cb = plt.colorbar(pc, ax=axes[-1], extend='max')
    cb.set_label(r"PMF ($k_{\mathrm{B}}T$)", rotation=-90, labelpad=10)
    fig.savefig(
        "/project/dinner/scguo/kaiB/dga/figures/pmfs.png", bbox_inches="tight"
    )


def main():
    temps = [87, 89, 91]
    hists_all = []
    for t in temps:
        hist = compute_hist(t)
        hists_all.append(hist)
    plot_pmf(hists_all)


if __name__ == "__main__":
    main()
