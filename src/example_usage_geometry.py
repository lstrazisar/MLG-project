from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import py3Dmol

def read_xyz(filename):
    """Read XYZ coordinates into a numpy array."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    n_atoms = int(lines[0])
    coords = []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        coords.append([float(x) for x in parts[1:4]])
    return np.array(coords)

smiles = "C(/C=C/c1ccc(-n2c3ccccc3c3ccccc32)cc1)=C\c1ccncc1"
smiles_modified = smiles.replace("/", "&").replace("\\", "$") # sanitize filenames
path = f"data/xyz/chromophores/{smiles_modified}.xyz"  #solvents if we have solvent

# Step 1: Create RDKit molecule from SMILES
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)

# Step 2: Read coordinates from XYZ file
coords = read_xyz(path)

# Step 3: Create a conformer and set coordinates
conf = Chem.Conformer(mol.GetNumAtoms())
for i, pos in enumerate(coords):
    conf.SetAtomPosition(i, pos)
mol.AddConformer(conf)

# AllChem.EmbedMolecule(mol)
# AllChem.UFFOptimizeMolecule(mol)
mb = Chem.MolToMolBlock(mol)

# Show with py3Dmol
view = py3Dmol.view(width=400, height=300)
view.addModel(mb, 'mol')
view.setStyle({'stick': {}})
view.zoomTo()
view.show()