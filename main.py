import numpy as np
from numpy.linalg import svd

import file_io as io


def get_residue_pos(resid, chain=''):
    """
    Method to return the positions of the atoms in a given residue

    ARGS:
    resid (int) :: residue ID given in the PDB file
    chain (str) :: chain name given in the PDB file

    returns:
    ndarray
    """

    if len(chain) > 0:
        res_out = np.array([atom[-1] for atom in my_atoms if (atom[0]==resid and atom[1]==chain)])
    else:
        res_out = np.array([atom[-1] for atom in my_atoms if atom[0]==resid])
        
    return res_out
    
    

if __name__ == '__main__':
    # Read in pdb file
    my_file = './h11-h4-rbd-simp.pdb'
    my_atoms = io.read_pdb(my_file)

    res = 490
    my_res = get_residue_pos(res, '')

    print(my_res)
