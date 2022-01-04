"""
pdb-plane-calc.file_io

AUTHOR: Neville B.-y. Yee
Date: 04-Jan-2022
"""

import Bio.PDB as pdb


def read_xyz(file_path):
    """
    Function to read in xyz file and output as list of atoms and positions

    ARGS:
    file_path (str): path to xyz file

    RETURNS:
    list
    """

    with open(file_path, "r") as f:
        xyz_file = f.readlines()[2:]


    atom_list = []
    for line in xyz_file:
        element, pos = line.split()[0], [float(x) for x in line.split()[1:]]
        atom_list.append([element, pos])

    return atom_list


def read_pdb(file_path):
    """
    Function to read in pdb file and output as list of atoms and positions

    ARGS:
    file_path (str): path to pdb file

    RETURNS:
    list
    """
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
                    pos = atom.get_coord()

                    # Fix element symbol casing issues
                    if len(element) > 1:
                        element = element[0].upper() + element[1:].lower()
                    atom_list.append([atom.get_full_id()[3][1],
                                      atom.get_full_id()[2],
                                      element,
                                      pos])

    return atom_list
