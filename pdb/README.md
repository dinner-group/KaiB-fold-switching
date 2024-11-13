# notes
`gs_d91r.pdb` is the "KaiB91R" mutant used for temperature experiments that Andy's group did.
It is the *Te* KaiB, residues 1-94, with mutations Y8A, D91R, and Y94A. This was obtained from
the ground state structure and mutated using CHARMM-GUI (Darren did this).

`fs_almut.pdb` is the construct Adam used for his original simulations. It is *Te* 1-99, with mutations
Y8A, G89A, D91R, and, Y94A. This is NOT the same as the "KaiB HDX" mutant that Andy's group used, since it
lacks the P71A mutation.

`fs_almut_p71a.pdb` is the same as above, but the P71A mutation. This is the same as the "KaiB HDX" construct
used for HDX experiments, and is what I (Spencer) used to run Upside HDX simulations.

`2qke_mon.pdb` and `fs_wtseq.pdb` are the wildtype *Te* sequences in the gs and fs structures, respectively. The 
2QKE corresponds to the dimeric gs found in the PDB, but with only one monomer, whereas the fs state was obtained 
from the mutated structure from Andy's old Science paper, but mutated *back to the wt sequence*. These were
also used by Adam for DGA simulations.

`remd_**.pdb` and `d91r_**.pdb` files are intermediates structures from REMD simulations of the D91R mutant used 
to seed more REMD simulations.
