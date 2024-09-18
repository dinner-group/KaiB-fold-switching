import scipy
import numpy as np
import numba as nb
from more_itertools import zip_equal

from extq.stop import forward_stop, backward_stop

def mjp_current(
    forward_q, backward_q, weights, in_domain, dtrajs, lag, N, normalize=True
):
    """Estimate the reactive current at each frame.

    Parameters
    ----------
    forward_q : list of (n_frames[i],) ndarray of float
        Forward committor for each frame.
    backward_q : list of (n_frames[i],) ndarray of float
        Backward committor for each frame.
    weights : list of (n_frames[i],) ndarray of float
        Reweighting factor to the invariant distribution for each frame.
    in_domain : list of (n_frames[i],) ndarray of bool
        Whether each frame of the trajectories is in the domain.
    dtrajs : list of (n_frames[i],) narray of int
        State index at each from
    lag : int
        Lag time in units of frames.
    N : int
        number of discrete states
    normalize : bool, optional
        If True (default), normalize `weights` to one.

    Returns
    -------
    (N, N) np.ndarray of float
        Reactive current for a Markov jump process between the discrete states

    """
    assert lag > 0
    out = 0.0
    for qp, qm, w, d, dtraj in zip_equal(
        forward_q, backward_q, weights, in_domain, dtrajs
    ):
        n_frames = w.shape[0]
        assert qp.shape == (n_frames,)
        assert qm.shape == (n_frames,)
        assert w.shape == (n_frames,)
        assert d.shape == (n_frames,)
        assert dtraj.shape == (n_frames,)
        assert np.all(w[max(0, n_frames - lag) :] == 0.0)
        if n_frames <= lag:
            j = np.zeros(len(w))
        else:
            tp = forward_stop(d)
            tm = backward_stop(d)
            out += _current_helper(qp, qm, w, tp, tm, dtraj, lag, N)
            # print(out)
    if normalize:
        wsum = sum(np.sum(w) for w in weights)
        for j in out:
            j /= wsum
    return out


@nb.njit
def _current_helper(qp, qm, w, tp, tm, dtraj, lag, N):
    n_frames = w.shape[0]
    result = np.zeros((n_frames, N, N))
    for start in range(len(w) - lag):
        end = start + lag
        for i in range(start, end):
            j = i + 1
            ti = max(tm[i], start)
            tj = min(tp[j], end)
            c = w[start] * qm[ti] * qp[tj] / lag
            result[i, dtraj[i], dtraj[j]] += 0.5 * c
            result[j, dtraj[i], dtraj[j]] += 0.5 * c
    return result.sum(axis=0)



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
    basis = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(len(dtrajs), nclusters))
    return basis
