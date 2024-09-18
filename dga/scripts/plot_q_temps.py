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
# mpl.rcParams["mathtext.fontset"] = "stixsans"

lag = 2000


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


def compute_hist(temp):
    base_dir = f"/project/dinner/scguo/kaiB/dga/{temp}"
    raw_feats, fs_qtots, f_rmsds, p_rmsds, r_rmsds = load_cvs(base_dir, 7, 32)
    c_green = [traj[3, :] - traj[2, :] for traj in fs_qtots]
    c_blue = [traj[5, :] - traj[4, :] for traj in fs_qtots]
    c_orange = [traj[1, :] - traj[0, :] for traj in fs_qtots]

    c_green_arr = np.concatenate(c_green)
    c_blue_arr = np.concatenate(c_blue)
    c_orange_arr = np.concatenate(c_orange)
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
    _, traj_inds = split_indices(raw_feats)
    in_domain = ~(in_fs | in_gs)
    in_d = np.split(in_domain, traj_inds)

    weights = np.load(f"{base_dir}/dga_data/weights.npy", allow_pickle=True)
    qp_fs2gs = np.load(f"{base_dir}/dga_data/qp_fs2gs.npy", allow_pickle=True)
    qp_gs2fs = np.load(f"{base_dir}/dga_data/qp_gs2fs.npy", allow_pickle=True)

    w_delay = [t[:-lag] for t in weights[-1]]
    cv1_delay = [t[:-lag] for t in c_green]
    cv2_delay = [t[:-lag] for t in c_blue]

    xe = np.linspace(-1.3, 1, 101)
    ye = np.linspace(-1, 1.3, 101)
    qp_delay = adjust_forward_committor(qp_fs2gs[4], in_d, lag)
    hist_fs2gs = extq.projection.average2d(
        cv1_delay, cv2_delay, qp_delay, w_delay, xe, ye
    )
    qp_delay = adjust_forward_committor(qp_gs2fs[4], in_d, lag)
    hist_gs2fs = extq.projection.average2d(
        cv1_delay, cv2_delay, qp_delay, w_delay, xe, ye
    )

    return hist_fs2gs, hist_gs2fs


def main():
    temps = [87, 89, 91]
    hists_fs2gs, hists_gs2fs = [], []
    for t in temps:
        hist_fs2gs, hist_gs2fs = compute_hist(t)
        hists_fs2gs.append(hist_fs2gs)
        hists_gs2fs.append(hist_gs2fs)

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

    # fs to gs committors
    for ax, hist in zip(axes, hists_fs2gs):
        pc = ax.pcolormesh(
            xc, yc, hist.T, cmap="nightfall", vmin=0, vmax=1, rasterized=True
        )
        ax.contour(xc, yc, hist.T, colors="#762A83", levels=[0.5], linewidths=1)
        ax.set_ylabel(
            r"$\alpha3_{\mathrm{fs}}\longleftrightarrow \beta4_{\mathrm{gs}}$"
        )
        ax.set_xlabel(r"$\beta4_{\mathrm{fs}}\longleftrightarrow\alpha3_{\mathrm{gs}}$")
        ax.label_outer()
    cb = plt.colorbar(pc, ax=axes[-1])
    cb.set_label(r"$q_{\mathrm{fs}\rightarrow\mathrm{gs}}$", rotation=-90, labelpad=10)
    fig.savefig(
        "/project/dinner/scguo/kaiB/dga/figures/qpfs2gs_tstate.pdf", bbox_inches="tight"
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(7, 2.25),
        dpi=500,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    # gs to fscommittors
    for ax, hist in zip(axes, hists_gs2fs):
        pc = ax.pcolormesh(
            xc, yc, hist.T, cmap="nightfall", vmin=0, vmax=1, rasterized=True
        )
        ax.contour(xc, yc, hist.T, colors="#762A83", levels=[0.5], linewidths=1)
        ax.set_ylabel(
            r"$\alpha3_{\mathrm{fs}}\longleftrightarrow \beta4_{\mathrm{gs}}$"
        )
        ax.set_xlabel(r"$\beta4_{\mathrm{fs}}\longleftrightarrow\alpha3_{\mathrm{gs}}$")
        ax.label_outer()
    cb = plt.colorbar(pc, ax=axes[-1])
    cb.set_label(r"$q_{\mathrm{gs}\rightarrow\mathrm{fs}}$", rotation=-90, labelpad=10)
    fig.savefig(
        "/project/dinner/scguo/kaiB/dga/figures/qpgs2fs_tstate.pdf", bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
