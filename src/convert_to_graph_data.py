import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from tqdm import tqdm

def atom_features(atom):
    """
    Calculate features for a single atom.
    You can add more features as needed.
    """
    return [
        atom.GetAtomicNum(),  # Atomic number
        atom.GetDegree(),  # Node degree
        atom.GetFormalCharge(),  # Formal charge
        atom.GetHybridization().real,  # Hybridization
        atom.GetIsAromatic(),  # Is aromatic
        atom.GetTotalNumHs(),  # Number of hydrogens
        atom.IsInRing(),  # Is in ring
    ]

def bond_features(bond):
    """
    Calculate features for a single bond.
    """
    return [
        bond.GetBondTypeAsDouble(),  # Bond type (1=single, 2=double, 3=triple)
        bond.GetIsConjugated(),  # Is conjugated
        bond.IsInRing(),  # Is in ring
    ]

def mol_to_graph(smiles):
    """
    Convert SMILES string to PyTorch Geometric Data object.
    
    Args:
        smiles: SMILES string of the molecule
    
    Returns:
        PyTorch Geometric Data object or None if SMILES is invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None
    
    # Add explicit hydrogens (optional)
    # mol = Chem.AddHs(mol)
    
    # Atom features (nodes)
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_features(atom))
    
    x = torch.tensor(atom_features_list, dtype=torch.float)
    
    # Edges - list of index pairs
    edge_index = []
    edge_attr = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Graphs are undirected, add both directions
        edge_index.append([i, j])
        edge_index.append([j, i])
        
        # Bond features
        bond_feat = bond_features(bond)
        edge_attr.append(bond_feat)
        edge_attr.append(bond_feat)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data

def create_dataset_from_csv(csv_path):
    """
    Read CSV and create dataset of graphs.
    Assumes you have columns: 'Chromophore' and 'Solvent' with SMILES strings.
    """
    df = pd.read_csv(csv_path)
    
    dataset = []
    
    for idx, row in tqdm(df.iterrows()):
        # Convert chromophore and solvent to graphs
        chromophore_graph = mol_to_graph(row['Chromophore'])
        solvent_graph = mol_to_graph(row['Solvent'])
        
        if chromophore_graph is None or solvent_graph is None:
            print(f"Warning: Row {idx} has invalid SMILES")
            continue
        
        # Target values (absorption and emission)
        y = torch.tensor([
            row['Absorption max (nm)'],
            row['Emission max (nm)']
        ], dtype=torch.float)
        
        # Store everything in dictionary
        sample = {
            'chromophore': chromophore_graph,
            'solvent': solvent_graph,
            'y': y,
            'tag': row['Tag']
        }
        
        dataset.append(sample)
    
    return dataset

# Example usage
if __name__ == "__main__":
    # Test example with benzene molecule
    # smiles = "CCN(CC)c1ccc2c(C(F)(F)F)cc(=O)oc2c1"
    # graph = mol_to_graph(smiles)
    
    # if graph is not None:
    #     print(f"Number of nodes (atoms): {graph.x.size(0)}")
    #     print(f"Number of edges: {graph.edge_index.size(1)}")
    #     print(f"Node feature dimension: {graph.x.size(1)}")
    #     print(f"\nFirst 5 node features:\n{graph.x[:5]}")
    
    # If you have CSV with SMILES strings:
    dataset = create_dataset_from_csv('data/splits/split_90_5_5/train.csv')
    print(f"Dataset size: {len(dataset)}")