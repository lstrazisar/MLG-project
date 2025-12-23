import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool, global_add_pool
from torch_geometric.nn.models import SchNet
from tqdm import tqdm

from src.utils import ChromophoreDataset, collate_fn


class DualGNN(nn.Module):
    """
    GNN model that processes chromophore and solvent separately,
    then combines their representations to predict absorption and emission.
    Supports multiple GNN architectures: GCN, GAT, GIN, SchNet.
    """

    def __init__(self, node_features=7, hidden_dim=64, output_dim=2, use_solvent=True, gnn_type='gcn', num_layers=2, use_descriptors=False, number_of_descriptors=205):
        """
        :param node_features: Number of input node features
        :param hidden_dim: Hidden dimension for GNN layers
        :param output_dim: Output dimension (2 for absorption and emission)
        :param use_solvent: Whether to use solvent information
        :param gnn_type: Type of GNN to use: 'gcn', 'gat', 'gin', 'schnet'
        :param num_layers: Number of GNN layers
        :param use_descriptors: Whether to use RDKit descriptors
        :param number_of_descriptors: Number of RDKit descriptors
        """
        super(DualGNN, self).__init__()

        self.use_solvent = use_solvent
        self.gnn_type = gnn_type
        self.use_descriptors = use_descriptors
        self.number_of_descriptors = number_of_descriptors

        # create GNN layers for chromophore
        self.chromo_convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = node_features if i == 0 else hidden_dim
            if gnn_type == 'gcn':
                self.chromo_convs.append(GCNConv(in_dim, hidden_dim))
            elif gnn_type == 'gat':
                heads = 4
                out_dim = hidden_dim // heads
                self.chromo_convs.append(GATConv(in_dim, out_dim, heads=heads, concat=True))
            elif gnn_type == 'gin':
                mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.chromo_convs.append(GINConv(mlp))
            elif gnn_type == 'schnet':
                self.chromo_convs.append(SchNet(hidden_channels=hidden_dim,
                                                num_filters=hidden_dim,
                                                num_interactions=num_layers))
                # SchNet handles all layers internally
                break

        if use_descriptors:
            self.descriptor_fc = nn.Linear(self.number_of_descriptors, hidden_dim * num_layers)  # equal neurons as GNN output
            if gnn_type == 'schnet':
                self.fc1 = nn.Linear(2, output_dim)
            else:
                self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)

        # for simplicity, we will not implement descriptor handling here
        elif use_solvent:
            # create GNN layers for solvent
            self.solvent_convs = nn.ModuleList()
            for i in range(num_layers):
                in_dim = node_features if i == 0 else hidden_dim
                if gnn_type == 'gcn':
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
                    # SchNet handles all layers internally
                    break

            # combine and predict
            if gnn_type == 'schnet':
                self.fc1 = nn.Linear(2, output_dim)
            else:
                self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)

        else:
            # only chromophore
            if gnn_type == 'schnet':
                self.fc1 = nn.Linear(1, output_dim)
            else:
                self.fc1 = nn.Linear(hidden_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(0.2)

        # batch normalization
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        if use_solvent:
            self.solvent_batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

    def forward(self, chromo_data, solvent_data, descriptor_data=None):
        """
        Forward pass through the model.

        :param chromo_data: PyG Data object for chromophore
        :param solvent_data: PyG Data object for solvent
        :param descriptor_data: Tensor of RDKit descriptors (if used)
        :return: Predicted absorption and emission properties
        """
        # process chromophore
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

        # global pooling
        if self.gnn_type == 'gin':
            x_c = global_add_pool(x_c, batch_c)  # GIN typically uses sum pooling
        elif self.gnn_type == 'schnet':
            x_c = x_c  # SchNet already outputs pooled representation
        else:
            x_c = global_mean_pool(x_c, batch_c)

        if self.use_descriptors:
            x = self.descriptor_fc(descriptor_data)
            x = F.relu(x)
            x = self.dropout(x)

        elif self.use_solvent:
            # process solvent
            x_s, edge_index_s, batch_s = solvent_data.x, solvent_data.edge_index, solvent_data.batch

            for i, conv in enumerate(self.solvent_convs):
                if self.gnn_type == 'schnet':
                    # SchNet only takes atomic numbers and positions
                    x_s = conv(x_s[:, 0].long(), solvent_data.pos, batch_s)
                else:
                    x_s = conv(x_s, edge_index_s)
                    x_s = self.solvent_batch_norms[i](x_s)
                    x_s = F.relu(x_s)
                    x_s = self.dropout(x_s)

            # global pooling
            if self.gnn_type == 'gin':
                x_s = global_add_pool(x_s, batch_s)  # GIN typically uses sum pooling
            elif self.gnn_type == 'schnet':
                x_s = x_s  # SchNet already outputs pooled representation
            else:
                x_s = global_mean_pool(x_s, batch_s)

            # combine
            x = torch.cat([x_c, x_s], dim=1)

        else:
            # only chromophore
            x = x_c

        if self.gnn_type == 'schnet':
            x = self.fc1(x)
        else:
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)

        return x


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    Returns average loss and MAE for absorption and emission.

    :param model: GNN model
    :param loader: DataLoader for training set
    :param optimizer: Optimizer
    :param criterion: Loss function
    :param device: Device to run training on
    :return: avg_loss, mae_absorption, mae_emission
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    for chromo_batch, solvent_batch, descriptor_data_list, targets in tqdm(loader, desc="Training"):
        # move data to device
        chromo_batch = chromo_batch.to(device)
        solvent_batch = solvent_batch.to(device)
        descriptor_data_list = descriptor_data_list.to(device) if descriptor_data_list is not None else None
        targets = targets.to(device)

        # forward pass
        optimizer.zero_grad()
        outputs = model(chromo_batch, solvent_batch, descriptor_data_list)
        loss = criterion(outputs, targets)
        # backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(outputs.detach().cpu())
        all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # calculate MAE for absorption and emission separately
    mae_absorption = torch.abs(all_preds[:, 0] - all_targets[:, 0]).mean().item()
    mae_emission = torch.abs(all_preds[:, 1] - all_targets[:, 1]).mean().item()

    return total_loss / len(loader), mae_absorption, mae_emission


def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on validation or test set.
    Returns average loss and MAE for absorption and emission.

    :param model: Trained GNN model
    :param loader: DataLoader for validation/test set
    :param criterion: Loss function
    :param device: Device to run evaluation on
    :return: avg_loss, mae_absorption, mae_emission
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for chromo_batch, solvent_batch, descriptor_data_list, targets in tqdm(loader, desc="Evaluating"):
            chromo_batch = chromo_batch.to(device)
            solvent_batch = solvent_batch.to(device)
            descriptor_data_list = descriptor_data_list.to(device) if descriptor_data_list is not None else None
            targets = targets.to(device)

            outputs = model(chromo_batch, solvent_batch, descriptor_data_list)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # calculate MAE for absorption and emission separately
    mae_absorption = torch.abs(all_preds[:, 0] - all_targets[:, 0]).mean().item()
    mae_emission = torch.abs(all_preds[:, 1] - all_targets[:, 1]).mean().item()

    return total_loss / len(loader), mae_absorption, mae_emission


def main():
    parser = argparse.ArgumentParser(description='Train GNN for chromophore property prediction')
    parser.add_argument('--data-dir', '-d', required=True, help='Directory containing train.csv, val.csv, test.csv')
    parser.add_argument('--output-dir', '-o', default='./models', help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--gnn-type', choices=['gcn', 'gat', 'gin', 'schnet'], default='gcn', help='Type of GNN: gcn, gat, gin or schnet')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--no-solvent', action='store_true', help='Use only chromophore (ignore solvent)')
    parser.add_argument('--use-descriptors', action='store_true', help='Use RDKit descriptors along with GNN for chromophore')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda or cpu)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.data_dir, 'train.csv')
    val_path = os.path.join(args.data_dir, 'val.csv')
    test_path = os.path.join(args.data_dir, 'test.csv')

    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    print("Loading datasets...")
    train_dataset = ChromophoreDataset(train_path, position=args.gnn_type == 'schnet', use_descriptors=args.use_descriptors)
    val_dataset = ChromophoreDataset(val_path, position=args.gnn_type == 'schnet', use_descriptors=args.use_descriptors)
    test_dataset = ChromophoreDataset(test_path, position=args.gnn_type == 'schnet', use_descriptors=args.use_descriptors)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device(args.device)
    model = DualGNN(
        hidden_dim=args.hidden_dim,
        use_solvent=not args.no_solvent,
        gnn_type=args.gnn_type,
        use_descriptors=args.use_descriptors,
        number_of_descriptors=train_dataset.descriptor_df.shape[1] if args.use_descriptors else 0,
        num_layers=args.num_layers
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'train_mae_abs': [], 'train_mae_em': [],
        'val_loss': [], 'val_mae_abs': [], 'val_mae_em': []
    }

    print(f"\nTraining on {device}")
    print(f"GNN type: {args.gnn_type.upper()}")
    print(f"Number of layers: {args.num_layers}")
    print(f"Model type: {'Chromophore only' if args.no_solvent else 'Chromophore + Solvent'}")
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    # training loop
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

        # save if best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args)
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print("Saved best model")

    # load best model and evaluate on test set
    print("\nLoading best model for test evaluation...")
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_mae_abs, test_mae_em = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE - Absorption: {test_mae_abs:.2f} nm, Emission: {test_mae_em:.2f} nm")

    # save training history
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # save test results
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
