import os
import sys
import glob

import numpy as np
import mdtraj as md


def main(index_file, outfile):
    xyz = []
    # assemble file list
    with open(index_file, mode="r") as f:
        for i, line in enumerate(f.readlines()):
            if i % 100 == 99:
                print(f"Finished loading frame {i + 1}")
            filename, frame = line.strip("\n").split()[:2]
            xyz.append(md.load_frame(filename, int(frame)).xyz)

    top = md.load_frame(filename, int(frame)).topology
    traj = md.Trajectory(np.concatenate(xyz), top)

    traj.save(outfile)


if __name__ == "__main__":
    index_file = sys.argv[1]
    outfile = sys.argv[2]

    if not os.path.exists(index_file):
        raise IOError(f"{index_file} does not exist")

    print(f"Index file: {index_file}")
    print(f"Output file: {outfile}")

    main(index_file, outfile)
