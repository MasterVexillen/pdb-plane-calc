import argparse
import numpy as np
from numpy.linalg import svd, norm

import file_io as io


def get_residue_pos(atom_list, resid, chain='', atoms=[]):
    """
    Method to return the positions of the atoms in a given residue

    ARGS:
    atom_list (list) :: list of atoms
    resid (int)      :: residue ID given in the PDB file
    chain (str)      :: chain name given in the PDB file
    atoms (list)     :: list of specified atoms

    returns:
    ndarray
    """

    subset = [atom for atom in atom_list if atom[0]==resid]

    if len(chain) > 0:
        subset = [atom for atom in subset if atom[1]==chain]

    if len(atoms) > 0:
        subset = [atom for atom in subset if atom[2].strip() in atoms]

    return np.array([atom[-1] for atom in subset])


def calc_plane_vector(atom_pos):
    """
    Method to calculate best-fitted (unit) plane vector given a set of points using SVD

    ARGS:
    atom_pos (ndarray) :: ndarray storing atomic positions

    returns:
    ndarray
    """
    
    # Zero-centering centroid of atoms before SVD
    atom_pos_0 = atom_pos.T - np.mean(atom_pos.T, axis=1, keepdims=True)
    u, v, sh = svd(atom_pos_0, full_matrices=True)

    return u[:, -1] / norm(u[:, -1])


def calc_plane_rmse(atom_pos, norm_vec):
    """
    Method to calculate RMSE of atoms from fitted plane

    ARGS:
    atom_pos (ndarray) :: ndarray storing atomic positions
    norm_vec (ndarray) :: normal vector of fitted plane

    returns:
    float
    """

    # Zero-centering centroid of atoms
    atom_pos_0 = atom_pos.T - np.mean(atom_pos.T, axis=1, keepdims=True)

    # Ensuring normal vector is a unit vector
    norm_vec /= norm(norm_vec)

    # Calculate distances and RMSE
    dist = np.dot(atom_pos_0.T, norm_vec)
    rmse = np.sqrt(np.sum(dist**2)/len(dist))

    return rmse
    

if __name__ == '__main__':
    # Argparse stuff from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("pdbfile",
                        type=str,
                        help="Path to the PDB file")
    parser.add_argument("resA",
                        type=int,
                        help="Residue ID of the first atom set")
    parser.add_argument("resB",
                        type=int,
                        help="Residue ID of the second atom set")
    parser.add_argument("--chainA",
                        type=str,
                        help="Chain name of the first atom set")
    parser.add_argument("--chainB",
                        type=str,
                        help="Chain name of the second atom set")
    parser.add_argument("--nameA",
                        type=str,
                        help="Atoms specified in set A")
    parser.add_argument("--nameB",
                        type=str,
                        help="Atoms specified in set B")
    

    args = parser.parse_args()

    my_file = args.pdbfile
    res_a_id = args.resA
    res_b_id = args.resB

    chain_a = ''
    if args.chainA is not None:
        chain_a = args.chainA

    chain_b = ''
    if args.chainB is not None:
        chain_b = args.chainB

    atoms_a = []
    if args.nameA is not None:
        atoms_a = [item.upper() for item in args.nameA.split(',')]

    atoms_b = []
    if args.nameB is not None:
        atoms_b = [item.upper() for item in args.nameB.split(',')]

    
    # Read in pdb file
    my_atoms = io.read_pdb(my_file)

    res_a_pos = get_residue_pos(my_atoms, res_a_id, chain_a, atoms_a)
    res_b_pos = get_residue_pos(my_atoms, res_b_id, chain_b, atoms_b)

    # Calculate normal vectors for atomic planes
    normal_vector_a = calc_plane_vector(res_a_pos)
    normal_vector_b = calc_plane_vector(res_b_pos)

    # Calculate RMSE of atoms on planes
    rmse_a = calc_plane_rmse(res_a_pos, normal_vector_a)
    rmse_b = calc_plane_rmse(res_b_pos, normal_vector_b)

    # Calculate angle between the two normal vectors
    angle = np.rad2deg(np.arccos(np.dot(normal_vector_a, normal_vector_b)))

    # Print out results
    print('\nGroup A coordinates:')
    print(res_a_pos)
    print('\nPlane A coefficients:')
    print(normal_vector_a)
    print('\nPlane A fitting RMSE:')
    print(rmse_a)
    

    print('\nGroup B coordinates:')
    print(res_b_pos)
    print('\nPlane B coefficients:')
    print(normal_vector_b)
    print('\nPlane B fitting RMSE:')
    print(rmse_b)

    print(f'\nAngle between groups: {angle}')
