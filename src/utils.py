import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm
import os


def atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetHybridization().real,
        atom.GetIsAromatic(),
        atom.GetTotalNumHs(),
        atom.IsInRing(),
    ]


def bond_features(bond):
    return [
        bond.GetBondTypeAsDouble(),
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]


def mol_to_graph(smiles, super_node=False, position=False, solvent=False):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    if mol is None:
        return None

    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_features(atom))

    x = torch.tensor(atom_features_list, dtype=torch.float)
    if super_node:
        super_node_feat = torch.tensor([[0]*7], dtype=torch.float)
        x = torch.cat([x, super_node_feat], dim=0)
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edge_index.append([i, j])
        edge_index.append([j, i])

        bond_feat = bond_features(bond)
        edge_attr.append(bond_feat)
        edge_attr.append(bond_feat)

    if super_node:
        super_idx = x.size(0) - 1
        for i in range(mol.GetNumAtoms()):
            edge_index.append([i, super_idx])
            edge_attr.append([0.0, 0.0, 0.0])

    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)

    if position:
        xyz_path = path_from_smiles(smiles, solvent=solvent)
        if not os.path.exists(xyz_path):
            return None
        else:
            coords = read_xyz(xyz_path)
            pos = torch.tensor(coords, dtype=torch.float)
            if pos.size(0) != x.size(0):
                # print(f"Position size {pos.size(0)} does not match number of atoms {x.size(0)} for SMILES: {smiles} for atoms {[atom.GetSymbol() for atom in mol.GetAtoms()]}")
                return None
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    else:
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


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


def path_from_smiles(smiles, solvent=False):
    smiles_modified = smiles.replace("/", "&").replace("\\", "$")
    if solvent:
        return f"data/xyz/solvents/{smiles_modified}.xyz"
    else:
        return f"data/xyz/chromophores/{smiles_modified}.xyz"


class ChromophoreDataset(Dataset):
    """Dataset for chromophore-solvent pairs"""

    def __init__(self, csv_path, super_node=False, position=False, use_descriptors=False):
        self.df = pd.read_csv(csv_path)
        self.valid_indices = []
        self.super_node = super_node
        self.position = position
        self.use_descriptors = use_descriptors
        # Pre-validate all SMILES
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            chromo_graph = mol_to_graph(row['Chromophore'], super_node=self.super_node, position=self.position)
            solvent_graph = mol_to_graph(row['Solvent'], super_node=self.super_node, position=self.position, solvent=True)

            if chromo_graph is not None and solvent_graph is not None:
                self.valid_indices.append(idx)

        if self.use_descriptors:
            self.descriptor_df = featurize_dataframe(self.df.iloc[self.valid_indices], smiles_column='Chromophore')
        
        print(f"Loaded {len(self.valid_indices)} valid samples from {len(self.df)} total")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        row = self.df.iloc[self.valid_indices[idx]]

        chromo_graph = mol_to_graph(row['Chromophore'], super_node=self.super_node, position=self.position)
        solvent_graph = mol_to_graph(row['Solvent'], super_node=self.super_node, position=self.position, solvent=True)
        descriptor_data = self.descriptor_df.iloc[idx].values if self.use_descriptors else None

        # Target values
        y = torch.tensor([
            row['Absorption max (nm)'],
            row['Emission max (nm)']
        ], dtype=torch.float)

        return chromo_graph, solvent_graph, descriptor_data, y


def collate_fn(batch):
    """Custom collate function for batching graphs"""
    chromo_graphs, solvent_graphs, descriptor_data_list, targets = zip(*batch)

    chromo_batch = Batch.from_data_list(chromo_graphs)
    solvent_batch = Batch.from_data_list(solvent_graphs)
    descriptor_data_list = torch.tensor(descriptor_data_list, dtype=torch.float) if descriptor_data_list[0] is not None else None
    targets = torch.stack(targets)

    return chromo_batch, solvent_batch, descriptor_data_list, targets


def compute_rdkit_descriptors(smiles):
    """
    Compute RDKit molecular descriptors for a SMILES string.
    Returns a dictionary of descriptor values.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Get all available descriptors
    descriptor_names = [desc[0] for desc in Descriptors.descList]
    descriptor_values = {}
    
    for name in descriptor_names:
        try:
            calc = getattr(Descriptors, name)
            descriptor_values[name] = calc(mol)
        except:
            descriptor_values[name] = np.nan
    
    return descriptor_values

def featurize_dataframe(df, smiles_column='Chromophore', valid_columns=None):
    """
    Convert SMILES strings to RDKit descriptor features.
    If valid_columns is provided, only keep those columns and fill missing with 0.
    """
    print(f"Computing RDKit descriptors for {len(df)} molecules...")
    
    descriptors_list = []
    valid_indices = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Featurizing molecules"):
        smiles = row[smiles_column]
        desc = compute_rdkit_descriptors(smiles)
        if desc is not None:
            descriptors_list.append(desc)
            valid_indices.append(idx)
        else:
            print(f"Warning: Invalid SMILES at index {idx}: {smiles}")
    
    # Create feature dataframe
    features_df = pd.DataFrame(descriptors_list, index=valid_indices)
    
    # Replace infinite values with NaN
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    
    if valid_columns is None:
        # Training phase: keep only valid columns (no NaN)
        features_df = features_df.dropna(axis=1)
        # drop column Ipc
        features_df = features_df.drop(columns=['Ipc'])
        valid_columns = features_df.columns.tolist()
        

        print(f"Generated {features_df.shape[1]} descriptors for {len(features_df)} valid molecules")
    else:
        # Val/Test phase: use only training columns, fill NaN with 0
        features_df = features_df[valid_columns].fillna(0)
        print(f"Using {features_df.shape[1]} descriptors (from training) for {len(features_df)} valid molecules")
    
    return features_df