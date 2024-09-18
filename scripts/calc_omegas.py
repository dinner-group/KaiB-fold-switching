import os, sys
import numpy as np
import tables as tb
import mdtraj as md

upside_path = os.environ["UPSIDE_HOME"]
upside_utils_dir = os.path.expanduser(upside_path + "/py")
sys.path.insert(0, upside_utils_dir)

import mdtraj_upside as mu
import upside_engine as ue

up_file   = sys.argv[1]
traj_file = sys.argv[2]

traj          = mu.load_upside_traj(traj_file)
pos           = mu.extract_bb_pos_angstroms(traj)[:]
engine        = ue.Upside(up_file)
n_frame       = traj.n_frames
diheds = []

for i in range(n_frame):
    engine.energy(pos[i])
    diheds.append(engine.get_output('Dihedral_OmegaTransCis')[:,0])

diheds = np.array(diheds)
np.save('{}_Omega.npy'.format(sys.argv[3]), diheds)
