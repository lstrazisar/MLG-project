import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from rdkit import Chem
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

    def __init__(self, csv_path, super_node=False, position=False):
        self.df = pd.read_csv(csv_path)
        self.valid_indices = []
        self.super_node = super_node
        self.position = position

        # Pre-validate all SMILES
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            chromo_graph = mol_to_graph(row['Chromophore'], super_node=self.super_node, position=self.position)
            solvent_graph = mol_to_graph(row['Solvent'], super_node=self.super_node, position=self.position, solvent=True)

            if chromo_graph is not None and solvent_graph is not None:
                self.valid_indices.append(idx)

        print(f"Loaded {len(self.valid_indices)} valid samples from {len(self.df)} total")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        row = self.df.iloc[self.valid_indices[idx]]

        chromo_graph = mol_to_graph(row['Chromophore'], super_node=self.super_node, position=self.position)
        solvent_graph = mol_to_graph(row['Solvent'], super_node=self.super_node, position=self.position, solvent=True)

        # Target values
        y = torch.tensor([
            row['Absorption max (nm)'],
            row['Emission max (nm)']
        ], dtype=torch.float)

        return chromo_graph, solvent_graph, y


def collate_fn(batch):
    """Custom collate function for batching graphs"""
    chromo_graphs, solvent_graphs, targets = zip(*batch)

    chromo_batch = Batch.from_data_list(chromo_graphs)
    solvent_batch = Batch.from_data_list(solvent_graphs)
    targets = torch.stack(targets)

    return chromo_batch, solvent_batch, targets
