import sys
import os
import numpy as np
import mdtraj as md

home_dir = "/project/dinner/scguo/kaiB"

upside_path = "/project/dinner/scguo/upside2/"
upside_utils_dir = os.path.expanduser(upside_path + "/py")
sys.path.insert(0, upside_utils_dir)
import mdtraj_upside as mu

def load_files(base_dir, n_s, n_i):
    files = []
    outputs = []
    for i in range(n_s):
        for j in range(n_i):
            for iso in ("cis", "trans"):
                idx = f"{i:02}_{j:02}_{iso}"
                head = f"{idx}_dga"
                if not os.path.exists(f"{base_dir}/{idx}/outputs/{head}_raw_feats.pkl"):
                    continue
                for k in range(2):
                    files.append(f"{base_dir}/{idx}/dga/{i:02}_{j:02}.run.{k:02}.up")
                    outputs.append(
                        f"{base_dir}/{idx}/outputs/{idx}_{k:02}_dssp.npy"
                    )
    return files, outputs


def convert_to_array(dssp, simplified=True):
    ans = np.zeros_like(dssp, dtype=int)
    if simplified:
        ans[np.where(dssp == "H")] = 0
        ans[np.where(dssp == "E")] = 1
        ans[np.where(dssp == "C")] = 2
    else:
        ans[np.where(dssp == "H")] = 0
        ans[np.where(dssp == "B")] = 1
        ans[np.where(dssp == "E")] = 2
        ans[np.where(dssp == "G")] = 3
        ans[np.where(dssp == "I")] = 4
        ans[np.where(dssp == "T")] = 5
        ans[np.where(dssp == "S")] = 6
        ans[np.where(dssp == " ")] = 7
    return ans


temp = int(sys.argv[1])
base_dir = f"/project/dinner/scguo/kaiB/dga/{temp}"
input_files, output_files = load_files(base_dir, 7, 32)
for inputf, outputf in zip(input_files, output_files):
    # traj = md.load(inputf)
    traj = mu.load_upside_traj(inputf)
    dssp = md.compute_dssp(traj, simplified=True)
    dssp = convert_to_array(dssp, simplified=True)
    np.save(outputf, dssp)
