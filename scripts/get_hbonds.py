import sys, os
import numpy as np
import tables as tb
import mdtraj as md

upside_path = os.environ['UPSIDE_HOME']
upside_utils_dir = os.path.expanduser(upside_path+"/py")
sys.path.insert(0, upside_utils_dir)

import mdtraj_upside as mu
import upside_engine as ue
import matplotlib.pyplot as plt

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('top_h5',     help='Input top file (.up or .h5 file)')
    parser.add_argument('input_h5',   help='Input simulation file')
    parser.add_argument('output_npy', help='Output npy file')
    parser.add_argument('--stride',   type=int, default=1, help='(default 1) Stride for reading file')
    parser.add_argument('--start',    type=int, default=0, help='(default 0) Initial frame')
    parser.add_argument('--residue',  type=str, default=None, help='(default none) the file used to store the residue id')
    parser.add_argument('--criterion1', type=float, default=0.01, help='(default 0.00) to judge whether NH is H-bonded. bigger than the criterion means H-bonded')
    args = parser.parse_args()

    engine = ue.Upside(args.top_h5)

    with tb.open_file(args.top_h5, 'r') as t:
        donor = t.root.input.potential.infer_H_O.donors.residue[:]
        n_donor = donor.size
        # for side chain buiral level
        weight = t.root.input.potential.sigmoid_coupling_environment.weights[:20]
        # for side chain-NH H-bond
        weight2 = weight*0
        weight2[3] = 1. # ASP
        weight2[6] = 1. # GLU

    traj    = mu.load_upside_traj(args.input_h5, top=args.top_h5)
    bb      = traj.top.select('name C or name N or name CA')
    traj_bb = traj.atom_slice(bb)[args.start::args.stride]
    N = traj_bb.n_frames

    print ("{} frames are used".format(N))

    Hbond1 = []
    for i in range(N):
        p = engine.energy(traj_bb.xyz[i]*10)

        hb  = engine.get_output('protein_hbond')[:,6]
        Hbond1.append(hb[:n_donor])

    Hbond1 = np.array(Hbond1)

    HB1 = Hbond1*0.
    HB1[Hbond1>args.criterion1] = 1.

    if '.npy' in args.output_npy:
        output_base = args.output_npy[:-4]
    else:
        output_base = args.output_npy[:]

    if args.residue:
        np.savetxt(args.residue, donor, fmt='%i')

    np.save(f"{output_base}_hb", HB1)

if __name__ == '__main__':
    main()
