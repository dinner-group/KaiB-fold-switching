import sys
import os

import numpy as np
import scipy
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

lag = 5000
names = ["blue", "green", "orange", "r_rmsd", "qp"]
labels = dict(
    blue=r"$\alpha3_{\mathrm{fs}}\longleftrightarrow \beta4_{\mathrm{gs}}$",
    green=r"$\beta4_{\mathrm{fs}}\longleftrightarrow\alpha3_{\mathrm{gs}}$",
    orange=r"$\beta3_{\mathrm{fs}}\longleftrightarrow\alpha2_{\mathrm{gs}}$",
    r_rmsd="core RMSD (nm)",
    qp=r"$q_+$",
)
lims = dict(blue=(-1, 1.3), green=(-1.3, 1), orange=(-1.3, 1), r_rmsd=(0, 1.5), qp=(0, 1))

plot = False
precomputed = False


def kdesum3d(
    x,
    y,
    z,
    w1,
    w2,
    w3,
    *,
    xmin=0,
    xmax=100,
    ymin=0,
    ymax=100,
    zmin=0,
    zmax=100,
    xstd=None,
    ystd=None,
    zstd=None,
    nx=50,
    ny=50,
    nz=50,
    cut=4.0,
    skip=1,
    **kwargs,
):
    """Compute a 3D kernel density estimate.

    This function histograms the data, then uses a Gaussian filter to
    approximate a kernel density estimate with a Gaussian kernel.

    Parameters
    ----------
    x, y, z : ndarray or list/tuple of ndarray
        Coordinates of each frame.
    w1, w2 : ndarray or list/tuple of ndarray
        Weight or value of each frame. The output is the sum of these
        values in each bin, after smoothing.
    xmin, xmax, ymin, ymax, zmin, zmax : float, optional
        Limits of kernel density estimate. If None, takes the min/max
        of the data along the coordinate.
    xstd, ystd, zstd : float, optional
        Standard deviation of the Gaussian filter. If None, these are
        set to (xmax - xmin) / nx and (ymax - ymin) / ny, respectively.
        Increase this to smooth the results more.
    nx, ny, nz : int, optional
        Number of bins in each dimension. This should be set as high as
        reasonable, since xstd/ystd takes care of the smoothing.
    cut : float, optional
        Number of standard deviations at which to truncate the Gaussian
        filter. The default, 4, usually doesn't need to be changed.

    Returns
    -------
    kde : (nx, ny) ndarray
        Kernel density estimate, given as bins.
    xedges : (nx+1,) ndarray
        Bin edges along the x dimension.
    yedges : (ny+1,) ndarray
        Bin edges along the y dimension.
    zedges: (nz+1,) ndarray
        Bin edges along the z dimension.

    """

    # flatten input to 1D arrays
    x = _flatten(np.asarray(x))
    y = _flatten(np.asarray(y))
    z = _flatten(np.asarray(z))
    w1 = _flatten(np.asarray(w1))
    w2 = _flatten(np.asarray(w2))
    w3 = _flatten(np.asarray(w3))

    # limits
    _xmin = np.min(x)
    _xmax = np.max(x)
    _ymin = np.min(y)
    _ymax = np.max(y)
    _zmin = np.min(z)
    _zmax = np.max(z)
    if xmin is None:
        xmin = _xmin
    if xmax is None:
        xmax = _xmax
    if ymin is None:
        ymin = _ymin
    if ymax is None:
        ymax = _ymax
    if zmin is None:
        zmin = _zmin
    if zmax is None:
        zmax = _zmax

    # separation between grid points
    xsep = (xmax - xmin) / nx
    ysep = (ymax - ymin) / ny
    zsep = (zmax - zmin) / nz

    # number of grid points to pad the boundaries,
    # since the Gaussian filter extends beyond the boundaries
    # usually overestimates the padding, but whatever
    ax = max(0, int(np.ceil((xmin - _xmin) / xsep + 1e-6)))
    bx = max(0, int(np.ceil((_xmax - xmax) / xsep + 1e-6)))
    ay = max(0, int(np.ceil((ymin - _ymin) / ysep + 1e-6)))
    by = max(0, int(np.ceil((_ymax - ymax) / ysep + 1e-6)))
    az = max(0, int(np.ceil((zmin - _zmin) / zsep + 1e-6)))
    bz = max(0, int(np.ceil((_zmax - zmax) / zsep + 1e-6)))

    # output bin edges
    xedges = np.linspace(xmin, xmax, nx + 1)
    yedges = np.linspace(ymin, ymax, ny + 1)
    zedges = np.linspace(zmin, zmax, nz + 1)

    # bin edges, with the added padding
    xedges_padded = np.concatenate(
        [
            xmin + xsep * np.arange(-ax, 0),
            xedges,
            xmax + xsep * np.arange(1, bx + 1),
        ]
    )
    yedges_padded = np.concatenate(
        [
            ymin + ysep * np.arange(-ay, 0),
            yedges,
            ymax + ysep * np.arange(1, by + 1),
        ]
    )
    zedges_padded = np.concatenate(
        [
            zmin + zsep * np.arange(-az, 0),
            zedges,
            zmax + zsep * np.arange(1, bz + 1),
        ]
    )
    assert np.allclose(xedges_padded[1:] - xedges_padded[:-1], xsep)
    assert np.allclose(yedges_padded[1:] - yedges_padded[:-1], ysep)
    assert np.allclose(zedges_padded[1:] - zedges_padded[:-1], zsep)
    assert xedges_padded[0] <= _xmin and _xmax <= xedges_padded[-1]
    assert yedges_padded[0] <= _ymin and _ymax <= yedges_padded[-1]
    assert zedges_padded[0] <= _zmin and _zmax <= zedges_padded[-1]

    # construct 3D histogram on padded edges
    hist_padded1, _ = np.histogramdd(
        (list(x), list(y), list(z)), weights=w1, bins=(xedges_padded, yedges_padded, zedges_padded)
    )

    # construct 3D histogram on padded edges
    hist_padded2, _ = np.histogramdd(
        (list(x), list(y), list(z)), weights=w2, bins=(xedges_padded, yedges_padded, zedges_padded)
    )

    # construct 3D histogram on padded edges
    hist_padded3, _ = np.histogramdd(
        (list(x), list(y), list(z)), weights=w3, bins=(xedges_padded, yedges_padded, zedges_padded)
    )

    # Gaussian kernel parameters
    if xstd is None:
        xstd = xsep
    if ystd is None:
        ystd = ysep
    if zstd is None:
        zstd = zsep

    # apply Gaussian filter to histogram
    kde_padded1 = scipy.ndimage.gaussian_filter(
        hist_padded1,
        sigma=(xstd / xsep, ystd / ysep, zstd / zsep),  # in units of grid points
        mode="constant",
        truncate=cut,
    )

    kde_padded2 = scipy.ndimage.gaussian_filter(
        hist_padded2,
        sigma=(xstd / xsep, ystd / ysep, zstd / zsep),  # in units of grid points
        mode="constant",
        truncate=cut,
    )
    kde_padded3 = scipy.ndimage.gaussian_filter(
        hist_padded3,
        sigma=(xstd / xsep, ystd / ysep, zstd / zsep),  # in units of grid points
        mode="constant",
        truncate=cut,
    )

    # remove the padding
    assert ax + nx + bx == kde_padded1.shape[0]
    assert ay + ny + by == kde_padded1.shape[1]
    assert az + nz + bz == kde_padded1.shape[2]
    kde1 = kde_padded1[ax : ax + nx, ay : ay + ny, az : az + nz]
    assert ax + nx + bx == kde_padded2.shape[0]
    assert ay + ny + by == kde_padded2.shape[1]
    assert az + nz + bz == kde_padded2.shape[2]
    kde2 = kde_padded2[ax : ax + nx, ay : ay + ny, az : az + nz]
    assert ax + nx + bx == kde_padded3.shape[0]
    assert ay + ny + by == kde_padded3.shape[1]
    assert az + nz + bz == kde_padded3.shape[2]
    kde3 = kde_padded3[ax : ax + nx, ay : ay + ny, az : az + nz]

    return kde1, kde2, kde3, xedges, yedges, zedges


def _flatten(a):
    if isinstance(a, np.ndarray):
        # avoid creating a new array (and using twice the memory)
        return np.ravel(a)
    else:
        return np.ravel(np.concatenate(a))


def load_cvs(base_dir, n_s, n_i):
    fs_qtots = []
    r_rmsds = []
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
                r_rmsds.extend(
                    np.load(
                        f"{base_dir}/{idx}/outputs/{head}_r_rmsds.pkl",
                        allow_pickle=True,
                    )
                )
    return fs_qtots, r_rmsds


def compute_hist(temp, cv1, cv2):
    base_dir = f"/project/dinner/scguo/kaiB/dga/{temp}"
    fs_qtots, r_rmsds = load_cvs(base_dir, 7, 32)
    c_green = [traj[3, :] - traj[2, :] for traj in fs_qtots]
    c_blue = [traj[5, :] - traj[4, :] for traj in fs_qtots]
    c_orange = [traj[1, :] - traj[0, :] for traj in fs_qtots]

    weights = np.load(f"{base_dir}/dga_data/weights.npy", allow_pickle=True)[-1]
    qp_fs2gs = np.load(f"{base_dir}/dga_data/qp_fs2gs.npy", allow_pickle=True)[4]
    j_qp = np.load(f"{base_dir}/dga_data/j_fs2gs_qp.npy")[0]
    j_blue = np.load(f"{base_dir}/dga_data/j_fs2gs_blue.npy")[0]
    j_green = np.load(f"{base_dir}/dga_data/j_fs2gs_green.npy")[0]
    j_orange = np.load(f"{base_dir}/dga_data/j_fs2gs_orange.npy")[0]
    j_r_rmsd = np.load(f"{base_dir}/dga_data/j_fs2gs_r_rmsds.npy")[0]

    cvs = dict(blue=c_blue, green=c_green, orange=c_orange, r_rmsd=r_rmsds, qp=qp_fs2gs)
    js = dict(blue=j_blue, green=j_green, orange=j_orange, r_rmsd=j_r_rmsd, qp=j_qp)

    nx, ny, nz = 50, 50, 10
    xmin, xmax = lims[cv1]
    ymin, ymax = lims[cv2]
    zmin, zmax = lims["qp"]

    std_bins = 4
    xstd = ((xmax - xmin) / nx) * std_bins
    ystd = ((ymax - ymin) / ny) * std_bins
    zstd = ((zmax - zmin) / nz) * 1

    xkde, ykde, zkde, xe, ye, ze = kdesum3d(
        cvs[cv1],
        cvs[cv2],
        cvs["qp"],
        js[cv1],
        js[cv2],
        js["qp"],
        nx=nx,
        ny=ny,
        nz=nz,
        xstd=xstd,
        ystd=ystd,
        zstd=zstd,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        zmin=zmin,
        zmax=zmax,
    )
    np.save(f"{base_dir}/dga_data/current_kde_{cv1}_{cv2}.npy", zkde)
    return xkde, ykde, zkde, xe, ye, ze


def plot_current_slice(kdes, xlim_flux, ylim_flux, cv1, cv2, index=4):
    fig, axes = plt.subplots(3, 2, figsize=(7, 2), sharex=True, sharey=True, constrained_layout=True, dpi=500)
    for (zkde, ax) in zip(kdes, axes.flat):
        qslice = zkde.swapaxes(2, 0)[index]
        pcm = ax.pcolormesh(
            xlim_flux, ylim_flux, qslice * 1e11, shading="flat", vmax=(np.mean(slice) + 3 * np.std(slice)), rasterized=True
        )
        cb = plt.colorbar(pcm, ax=ax)
        cb.set_label(r"$J\cdot \nabla q_{\mathrm{fs}\rightarrow\mathrm{gs}}\times 10^{-11}$", rotation=-90, labelpad=10)
        ax.set_xlabel(labels[cv1])
        ax.set_ylabel(labels[cv2])
        ax.label_outer()
    fig.savefig(
        f"/project/dinner/scguo/kaiB/dga/figures/currents_slice{index}.png", bbox_inches="tight"
    )


def main():
    cv1 = sys.argv[1]
    cv2 = sys.argv[2]
    if not (cv1 in names and cv2 in names):
        raise KeyError
    q_ind = int(sys.argv[3])

    temps = [87, 89, 91]
    hists_all = []
    for t in temps:
        if not precomputed:
            xkde, ykde, zkde, xe, ye, ze = compute_hist(t, cv1, cv2)
        else:
            base_dir = f"/project/dinner/scguo/kaiB/dga/{temp}"
            zkde = np.load(f"{base_dir}/dga_data/current_kde_{cv1}_{cv2}.npy")
        hists_all.append(zkde)
    if plot:
        plot_pmf(hists_all, xe, ye, cv1, cv2, index=q_index)


if __name__ == "__main__":
    main()
