import argparse
import numpy as np
from numpy.linalg import svd, norm
from scipy.spatial.transform import Rotation as R
import matplotlib.path as mpltPath


def read_pdb(file_path):
    """
    Function to read in pdb file and output as list of atoms and positions

    ARGS:
    file_path (str): path to pdb file

    RETURNS:
    list
    """
    import Bio.PDB as pdb
    from Bio.PDB.PDBExceptions import PDBConstructionWarning
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PDBConstructionWarning)
        
    parser = pdb.PDBParser()
    structure = parser.get_structure("MySample", file_path)

    # Extract data
    atom_list = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    element = atom.element
                    # Fix element symbol casing issues
                    if len(element) > 1:
                        element = element[0].upper() + element[1:].lower()
                    atom_list.append([atom.get_full_id()[3][1],
                                      atom.get_full_id()[2],
                                      atom.fullname,
                                      element,
                                      atom.get_coord()])

    return atom_list


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
        subset = sorted(subset, key=lambda x: atoms.index(x[2].strip()))

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

    # Obtain unit plane vector and ensure it points upwards (z>0)
    unit_n = u[:, -1] / norm(u[:, -1])

    return unit_n * np.sign(unit_n[-1])


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
    atom_pos_0 = atom_pos - np.mean(atom_pos, axis=0, keepdims=True)

    # Ensuring normal vector is a unit vector
    norm_vec /= norm(norm_vec)

    # Calculate distances and RMSE
    dist = np.dot(atom_pos_0, norm_vec)
    rmse = np.sqrt(np.sum(dist**2)/len(dist))

    return rmse


def calc_rotation_vector(orig, target):
    """
    Method to calculate rotation vector

    ARGS:
    orig (ndarray) :: original vector to be rotated
    target (ndarray) :: target vector

    Returns:
    ndarray
    """

    # Make sure the given vectors are unit vectors
    orig /= norm(orig)
    target /= norm(target)

    rot_axis = np.cross(orig, target) / norm(np.cross(orig, target))
    rot_angle = np.arccos(np.dot(orig, target))

    return rot_axis * rot_angle


def rotate_points(points, rot_vec, preserve_centroid=False):
    """
    Method to rotate points with given rotation vector

    ARGS:
    points (ndarray) :: ndarray storing atomic positions
    rot_vec (ndarray) :: rotation vector with which to rotate the system
    preserve_centroid (boolean) :: whether to rotate wrt origin or wrt centroid of system

    Returns:
    ndarray
    """

    points_centroid = np.mean(points, axis=0, keepdims=True)
    r = R.from_rotvec(rot_vec)

    if preserve_centroid:
        points_origin = points - points_centroid
        points_origin = r.apply(points_origin)
        new_points = points_origin + points_centroid
    else:
        new_points = r.apply(points)

    return new_points


def generate_mc_points(contour_pts, num_points):
    # Determine boundaries
    xmin = min(contour_pts[:, 0])
    ymin = min(contour_pts[:, 1])
    xmax = max(contour_pts[:, 0])
    ymax = max(contour_pts[:, 1])

    # Generate points
    xx = np.random.uniform(xmin, xmax, size=(num_points, 1))
    yy = np.random.uniform(ymin, ymax, size=(num_points, 1))

    return(np.concatenate((xx, yy), axis=1))


def points_in_polygon(points, contour):
    my_path = mpltPath.Path(contour[:, :2])
    points_inside = my_path.contains_points(points[:, :2])

    return np.array(points_inside)


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
    parser.add_argument("-np", "--num_points",
                        type=int,
                        help="Number of points in Monte Carlo sampling (Default: 1e6)")
    parser.add_argument("-e", "--extra_info",
                        action="store_true",
                        help="Outputs extra info (subset coordinates, plane vectors)")

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

    numpoints = int(1e6)
    if args.num_points is not None:
        numpoints = int(args.num_points)


    # read in pdb file
    my_atoms = read_pdb(my_file)

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

    # Print out results (if with flag)
    if args.extra_info:
        print('\nGroup A coordinates:')
        print(res_a_pos)
        print('\nPlane A coefficients:')
        print(normal_vector_a)

        print(f'\n------------------------------')
        print('\nGroup B coordinates:')
        print(res_b_pos)
        print('\nPlane B coefficients:')
        print(normal_vector_b)

        print(f'\n------------------------------')


    # CALCULATE COEFFICIENT OF OVERLAP
    # Calculate rotation vectors for the planes
    rot_vec_a = calc_rotation_vector(normal_vector_a, np.array([0, 0, 1], dtype=float))
    rot_vec_b = calc_rotation_vector(normal_vector_b, np.array([0, 0, 1], dtype=float))

    # Move centroid of system to origin
    whole_system_centroid = np.mean(np.concatenate((res_a_pos, res_b_pos)), axis=0)
    res_a_trans = res_a_pos - whole_system_centroid
    res_b_trans = res_b_pos - whole_system_centroid

    # Rotate the residues with respect to Residue A
    res_a_trans_rot = rotate_points(res_a_trans, rot_vec_a)
    res_b_trans_rot = rotate_points(res_b_trans, rot_vec_a)

    # Generate MC sampling points
    mc_contour = np.concatenate((res_a_trans_rot, res_b_trans_rot))
    mc_samples = generate_mc_points(mc_contour, numpoints)

    # Calculate maximum overlapping area (in number of points)
    res_a_orig_rot = res_a_trans_rot - np.mean(res_a_trans_rot, axis=0)
    res_b_orig_rot = res_b_trans_rot - np.mean(res_b_trans_rot, axis=0)
    points_in_a_orig = points_in_polygon(mc_samples, res_a_orig_rot)
    points_in_b_orig = points_in_polygon(mc_samples, res_b_orig_rot)
    max_overlap_area = np.sum(np.logical_and(points_in_a_orig, points_in_b_orig))

    # Calculate real overlapping area (in number of points)
    points_in_a = points_in_polygon(mc_samples, res_a_trans_rot)
    points_in_b = points_in_polygon(mc_samples, res_b_trans_rot)
    overlap_area = np.sum(np.logical_and(points_in_a, points_in_b))

    # Calculate and output coefficient of overlap
    coeff_overlap = overlap_area / max_overlap_area

    # Calculate and output coplanar distance between centroids of two groups
    centroid_a = np.mean(res_a_trans_rot, axis=0)
    centroid_b = np.mean(res_b_trans_rot, axis=0)
    coplanar_dist = norm(centroid_a[:2] - centroid_b[:2])

    # Outputting essential results
    print('\nPlane A fitting RMSE:')
    print(f'{rmse_a} A')
    print('\nPlane B fitting RMSE:')
    print(f'{rmse_b} A')
    
    print(f'\nAngle between groups: {angle: 7.5f} degs')
    print(f'Coplanar distance: {coplanar_dist: 7.5f} A')
    print(f'\nCoefficient of Overlap: {coeff_overlap: 10.5f} \n')

    print('ALL CALCULATIONS FINISHED...')
    print(f'------------------------------\n')
