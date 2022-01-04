import argparse
import numpy as np
from numpy.linalg import svd

import file_io as io


def get_residue_pos(atom_list, resid, chain=''):
    """
    Method to return the positions of the atoms in a given residue

    ARGS:
    atom_list (list) :: list of atoms
    resid (int)      :: residue ID given in the PDB file
    chain (str)      :: chain name given in the PDB file

    returns:
    ndarray
    """

    if len(chain) > 0:
        res_out = np.array([atom[-1] for atom in atom_list if (atom[0]==resid and atom[1]==chain)])
    else:
        res_out = np.array([atom[-1] for atom in atom_list if atom[0]==resid])
        
    return res_out


    
    

if __name__ == '__main__':
    # Argparse stuff from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("pdbfile",
                        type=str,
                        help="Path to the PDB file")
    parser.add_argument("resA",
                        type=int,
                        help="Residue ID of the first atom set")
    parser.add_argument("--chainA",
                        type=str,
                        help="Chain name of the first atom set")
    parser.add_argument("resB",
                        type=int,
                        help="Residue ID of the second atom set")
    parser.add_argument("--chainB",
                        type=str,
                        help="Chain name of the second atom set")

    args = parser.parse_args()

    my_file = args.pdbfile
    res_a_id = args.resA
    res_b_id = args.resB

    if args.chainA is not None:
        chain_a = args.chainA
    else:
        chain_a = ''

    if args.chainB is not None:
        chain_b = args.chainB
    else:
        chain_b = ''

    
    # Read in pdb file
    my_atoms = io.read_pdb(my_file)

    res_a_pos = np.array(get_residue_pos(my_atoms, res_a_id, chain_a))
    res_b_pos = np.array(get_residue_pos(my_atoms, res_b_id, chain_b))

