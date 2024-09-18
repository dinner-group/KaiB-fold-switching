import sys
import os
import glob
import numpy as np
import mdtraj as md
from joblib import Memory

home_dir = "/project/dinner/scguo/kaiB"
upside_path = "/project/dinner/scguo/upside2/"
upside_utils_dir = os.path.expanduser(upside_path + "/py")
sys.path.insert(0, upside_utils_dir)
import mdtraj_upside as mu

temp = int(sys.argv[1])
base_dir = f"{home_dir}/dga/new_{temp}"
data_dir = f"{base_dir}/data"

memory = Memory(data_dir)

def load_files(base_dir):
    traj_files, outputs = [], []
    for i in range(1, 12289):
        j = i // 1000
        for iso in ("cis", "trans"):
            head = f"{base_dir}/{j:02}/{i:05}/outputs/{i:05}_{iso}"
            listed = glob.glob(f"{base_dir}/{j:02}/{i:05}/{iso}/*.h5")
            if len(listed) != 0:
                traj_files.extend(listed)
                outputs.append(f"{head}_dssp.npy")
    return traj_files, outputs


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


@memory.cache
def compute_dssp(trajfile):
    traj = md.load(trajfile)
    dssp = md.compute_dssp(traj, simplified=True)
    dssp = convert_to_array(dssp, simplified=True)
    return dssp


input_files, output_files = load_files(base_dir)
for inputf, outputf in zip(input_files, output_files):
    dssp = compute_dssp(inputf)
    np.save(outputf, dssp)
