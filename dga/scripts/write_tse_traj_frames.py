import os
import numpy as np

home_dir = "/project/dinner/scguo/kaiB"


def bin_inds(q, qstep=0.05, low=0, hi=1):
    q_arr = np.ravel(q)
    # print(q_arr.shape)
    nsteps = round((hi - low) / qstep)
    all_inds = []
    steps = np.linspace(low, hi - qstep, nsteps)
    for i, s in enumerate(steps):
        q_inds = ((q_arr >= s) & (q_arr <= s + qstep)).nonzero()[0]
        all_inds.append(q_inds)
    return steps, all_inds


def load_filenames(base_dir, n_s, n_i):
    filenames = []
    for i in range(n_s):
        for j in range(n_i):
            for iso in ("cis", "trans"):
                idx = f"{i:02}_{j:02}_{iso}"
                for k in range(2):
                    filenames.append(
                        f"{base_dir}/{idx}/dga/{i:02}_{j:02}.run.{k:02}.h5"
                    )
    return filenames


def save_frames(qp, weights, q_bin_dir, filenames, n=1000):
    steps, q_inds = bin_inds(qp, qstep=0.2, low=0.4, hi=0.6)
    qp_arr = np.ravel(qp)
    w_arr = np.ravel(weights)
    q_inds = q_inds[0]
    print(len(q_inds))

    #weight
    w = w_arr[q_inds]
    w /= np.sum(w)
    all_top = np.random.choice(q_inds, n * 10, replace=False, p=w)
    for i, top in enumerate(np.split(all_top, 10)):
        # print(top)
        with open(f"{q_bin_dir}/q_tse_{i}.txt", mode="w") as f:
            for close_id in top:
                traj_ind, frame_ind = divmod(close_id, 39997)
                # print(close_id, traj_ind, frame_ind)
                f.write(f"{filenames[traj_ind]}\t{frame_ind}\t{close_id}\t{qp_arr[close_id]}\n")


for temp in (87, 89, 91):
    base_dir = f"/project/dinner/scguo/kaiB/dga/{temp}"
    qp_gs2fs = np.load(f"{base_dir}/dga_data/qp_gs2fs_lag1000_mem4.npy", allow_pickle=True)
    weights = np.load(f"{base_dir}/dga_data/weights.npy")[4]
    q_bin_dir = os.path.join(base_dir, "q_bin")
    if not os.path.exists(q_bin_dir):
        os.mkdir(q_bin_dir)
    filenames = load_filenames(base_dir, 7, 32)
    save_frames(qp_gs2fs, weights, q_bin_dir, filenames)
