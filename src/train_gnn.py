import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
from torch_geometric.nn.models import SchNet 
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from rdkit import Chem
import argparse
import os
from tqdm import tqdm
import json

# Import the mol_to_graph function from previous script
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

def mol_to_graph(smiles,super_node=False,position= False, solvent=False):
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

def path_from_smiles(smiles,solvent=False):
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


class DualGNN(nn.Module):
    """
    GNN model that processes chromophore and solvent separately,
    then combines their representations to predict absorption and emission.
    Supports multiple GNN architectures: GCN, GAT, GIN
    """
    
    def __init__(self, node_features=7, hidden_dim=64, output_dim=2, use_solvent=True, gnn_type='gcn', num_layers=2):
        super(DualGNN, self).__init__()
        
        self.use_solvent = use_solvent
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        
        # Create GNN layers for chromophore
        self.schnet_chromo = None
        self.chromo_convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = node_features if i == 0 else hidden_dim
            if gnn_type == 'gcn' or gnn_type == 'gcn+super_node':
                self.chromo_convs.append(GCNConv(in_dim, hidden_dim))
            elif gnn_type == 'gat':
                # GAT with 4 attention heads
                heads = 4
                out_dim = hidden_dim // heads
                self.chromo_convs.append(GATConv(in_dim, out_dim, heads=heads, concat=True))
            elif gnn_type == 'gin':
                # GIN with MLP
                mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.chromo_convs.append(GINConv(mlp))
            elif gnn_type == 'schnet':
                # SchNet
                self.chromo_convs.append(SchNet(hidden_channels=hidden_dim,
                                        num_filters=hidden_dim,
                                        num_interactions=num_layers))
                break
        
        if use_solvent:
            # Create GNN layers for solvent
            self.solvent_convs = nn.ModuleList()
            for i in range(num_layers):
                in_dim = node_features if i == 0 else hidden_dim
                if gnn_type == 'gcn' or gnn_type == 'gcn+super_node':
                    self.solvent_convs.append(GCNConv(in_dim, hidden_dim))
                elif gnn_type == 'gat':
                    heads = 4
                    out_dim = hidden_dim // heads
                    self.solvent_convs.append(GATConv(in_dim, out_dim, heads=heads, concat=True))
                elif gnn_type == 'gin':
                    mlp = nn.Sequential(
                        nn.Linear(in_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim)
                    )
                    self.solvent_convs.append(GINConv(mlp))
                elif gnn_type == 'schnet':
                    self.solvent_convs.append(SchNet(hidden_channels=hidden_dim,
                                        num_filters=hidden_dim,
                                        num_interactions=num_layers))
                    break
            
            # Combine and predict
            if gnn_type == 'schnet':
                self.fc1 = nn.Linear(2, output_dim)
            else:
                self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            # Only chromophore
            if gnn_type == 'schnet':
                self.fc1 = nn.Linear(1, output_dim)
            else:
                self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        # if gnn_type == 'schnet':
        #     self.fc2 = nn.Linear(1, output_dim)
        # else:
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(0.3)
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        if use_solvent:
            self.solvent_batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
    
    def forward(self, chromo_data, solvent_data):
        # Process chromophore
        x_c, edge_index_c, batch_c = chromo_data.x, chromo_data.edge_index, chromo_data.batch
        for i, conv in enumerate(self.chromo_convs):
            if self.gnn_type == 'schnet':
                # SchNet only takes atomic numbers and positions
                x_c = conv(x_c[:, 0].long(), chromo_data.pos, batch_c)
            else:
                x_c = conv(x_c, edge_index_c)
                x_c = self.batch_norms[i](x_c)
                x_c = F.relu(x_c)
                x_c = self.dropout(x_c)
        

            

        # Global pooling
        if self.gnn_type == 'gcn+super_node':
            indexes = np.append(np.where(np.diff(batch_c) != 0)[0], len(batch_c)-1)
            x_c = x_c[indexes]
        elif self.gnn_type == 'gin':
            x_c = global_add_pool(x_c, batch_c)  # GIN typically uses sum pooling
        elif self.gnn_type == 'schnet':
            x_c = x_c  # SchNet already outputs pooled representation
        else:
            x_c = global_mean_pool(x_c, batch_c)
        
        if self.use_solvent:
            # Process solvent
            x_s, edge_index_s, batch_s = solvent_data.x, solvent_data.edge_index, solvent_data.batch
            
            for i, conv in enumerate(self.solvent_convs):
                if self.gnn_type == 'schnet':
                    x_s = conv(x_s[:, 0].long(), solvent_data.pos, batch_s)
                else:
                    x_s = conv(x_s, edge_index_s)
                    x_s = self.solvent_batch_norms[i](x_s)
                    x_s = F.relu(x_s)
                    x_s = self.dropout(x_s)
            

            # Global pooling
            if self.gnn_type == 'gcn+super_node':
                indexes = np.append(np.where(np.diff(batch_s) != 0)[0], len(batch_s)-1)
                x_s = x_s[indexes]
            elif self.gnn_type == 'gin':
                x_s = global_add_pool(x_s, batch_s)
            elif self.gnn_type == 'schnet':
                x_s = x_s
            else:
                x_s = global_mean_pool(x_s, batch_s)
            
            # Combine
            x = torch.cat([x_c, x_s], dim=1)
        else:
            # Only chromophore
            x = x_c
        if self.gnn_type == 'schnet':
            x = self.fc1(x)
        else:
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
        
        return x


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for chromo_batch, solvent_batch, targets in tqdm(loader, desc="Training"):
        chromo_batch = chromo_batch.to(device)
        solvent_batch = solvent_batch.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(chromo_batch, solvent_batch)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.append(outputs.detach().cpu())
        all_targets.append(targets.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate MAE for absorption and emission
    mae_absorption = torch.abs(all_preds[:, 0] - all_targets[:, 0]).mean().item()
    mae_emission = torch.abs(all_preds[:, 1] - all_targets[:, 1]).mean().item()
    
    return total_loss / len(loader), mae_absorption, mae_emission


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for chromo_batch, solvent_batch, targets in tqdm(loader, desc="Evaluating"):
            chromo_batch = chromo_batch.to(device)
            solvent_batch = solvent_batch.to(device)
            targets = targets.to(device)
            
            outputs = model(chromo_batch, solvent_batch)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate MAE for absorption and emission separately
    mae_absorption = torch.abs(all_preds[:, 0] - all_targets[:, 0]).mean().item()
    mae_emission = torch.abs(all_preds[:, 1] - all_targets[:, 1]).mean().item()
    
    return total_loss / len(loader), mae_absorption, mae_emission


def main():
    parser = argparse.ArgumentParser(description='Train GNN for chromophore property prediction')
    parser.add_argument('--data-dir', '-d', required=True, help='Directory containing train.csv, val.csv, test.csv')
    parser.add_argument('--output-dir', '-o', default='./models', help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--gnn-type', choices=['gcn', 'gat', 'gin','gcn+super_node', 'schnet'], default='gcn', help='Type of GNN: gcn, gat, gin, gcn+super_node or schnet')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--no-solvent', action='store_true', help='Use only chromophore (ignore solvent)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Construct paths to split files
    train_path = os.path.join(args.data_dir, 'train.csv')
    val_path = os.path.join(args.data_dir, 'val.csv')
    test_path = os.path.join(args.data_dir, 'test.csv')
    
    # Check if files exist
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = ChromophoreDataset(train_path, super_node=args.gnn_type == 'gcn+super_node', position=args.gnn_type == 'schnet')
    val_dataset = ChromophoreDataset(val_path, super_node=args.gnn_type == 'gcn+super_node', position=args.gnn_type == 'schnet')
    test_dataset = ChromophoreDataset(test_path, super_node=args.gnn_type == 'gcn+super_node', position=args.gnn_type == 'schnet')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    device = torch.device(args.device)
    model = DualGNN(
        hidden_dim=args.hidden_dim, 
        use_solvent=not args.no_solvent,
        gnn_type=args.gnn_type,
        num_layers=args.num_layers
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'train_mae_abs': [], 'train_mae_em': [],
        'val_loss': [], 'val_mae_abs': [], 'val_mae_em': []
    }
    
    print(f"\nTraining on {device}")
    print(f"GNN type: {args.gnn_type.upper()}")
    print(f"Number of layers: {args.num_layers}")
    print(f"Model: {'Chromophore only' if args.no_solvent else 'Chromophore + Solvent'}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_mae_abs, train_mae_em = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae_abs, val_mae_em = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_mae_abs'].append(train_mae_abs)
        history['train_mae_em'].append(train_mae_em)
        history['val_loss'].append(val_loss)
        history['val_mae_abs'].append(val_mae_abs)
        history['val_mae_em'].append(val_mae_em)
        
        print(f"Train Loss: {train_loss:.4f} | MAE - Absorption: {train_mae_abs:.2f} nm, Emission: {train_mae_em:.2f} nm")
        print(f"Val Loss: {val_loss:.4f} | MAE - Absorption: {val_mae_abs:.2f} nm, Emission: {val_mae_em:.2f} nm")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args)
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print("✓ Saved best model")
    
    # Load best model and evaluate on test set
    print("\nLoading best model for test evaluation...")
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_mae_abs, test_mae_em = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE - Absorption: {test_mae_abs:.2f} nm, Emission: {test_mae_em:.2f} nm")
    
    # Save training history
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save test results
    test_results = {
        'test_loss': test_loss,
        'test_mae_absorption': test_mae_abs,
        'test_mae_emission': test_mae_em
    }
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()