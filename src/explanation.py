# example run:
# python3 -m src.explanation --data-dir ./data/splits/scaffold_split_80_10_10/ --method atom

import argparse
import os
import torch
from torch_geometric.data import Data, Batch

from src.utils import ChromophoreDataset
from src.train_gnn import DualGNN
from src.draw import draw_mol_importance, collapse_H_to_heavy, draw_bond_importance


def _remove_bond_edges_pyg(data: Data, a: int, b: int) -> Data:
    """
    Remove both directed edges a->b and b->a (and corresponding edge_attr rows if present).

    :param data: PyG Data object
    :param a: node index of one end of the bond
    :param b: node index of the other end of the bond
    :return: new Data object with the bond removed
    """
    out = data.clone()
    ei = out.edge_index
    mask = ~(((ei[0] == a) & (ei[1] == b)) | ((ei[0] == b) & (ei[1] == a)))
    out.edge_index = ei[:, mask]
    if hasattr(out, "edge_attr") and out.edge_attr is not None and out.edge_attr.numel() > 0:
        out.edge_attr = out.edge_attr[mask]
    return out


@torch.no_grad()
def bond_removal_attribution(model, chromo_data, solvent_data, device):
    """
    Compute bond importance by removing each bond and measuring change in prediction.

    :param model: GNN model
    :param chromo_data: Data object for chromophore
    :param solvent_data: Data object for solvent or None
    :param device: torch device
    :return: base prediction tensor shape [2], list of ((a,b), importance tensor shape [2]) for each bond
    """
    model.eval()
    chromo_b = Batch.from_data_list([chromo_data]).to(device)
    solvent_b = Batch.from_data_list([solvent_data]).to(device) if solvent_data is not None else None
    base = model(chromo_b, solvent_b).squeeze(0).detach().cpu()

    seen = set()
    bond_imp = []
    # iterate undirected bonds from directed edge_index pairs
    ei = chromo_data.edge_index
    for k in range(ei.size(1)):
        a = int(ei[0, k])
        b = int(ei[1, k])
        if (b, a) in seen or (a, b) in seen:
            continue
        seen.add((a, b))
        chromo2 = _remove_bond_edges_pyg(chromo_data.to(device), a, b)
        pred2 = model(Batch.from_data_list([chromo2]).to(device), solvent_b).squeeze(0).detach().cpu()
        bond_imp.append(((a, b), base - pred2))
    return base, bond_imp


def _mask_atom_schnet(data: Data, node_idx: int, z_mask: int = 1) -> Data:
    """
    For SchNet: replace atomic number at node_idx with z_mask (default H=1).
    Keeps graph size + pos (much less OOD than deletion).
    Assumes atomic number stored in x[:,0].

    :param data: PyG Data object
    :param node_idx: index of node to mask
    :param z_mask: atomic number to use for masking
    :return: new Data object with masked atom
    """
    out = data.clone()
    out.x = out.x.clone()
    out.x[node_idx, 0] = float(z_mask)
    return out


@torch.no_grad()
def atom_mask_attribution_schnet(model, chromo_data, solvent_data, device, z_mask=1, explain_solvent=True):
    """
    Compute atom importance by masking each atom (setting atomic number to z_mask) and measuring change in prediction.

    :param model: GNN model
    :param chromo_data: Data object for chromophore
    :param solvent_data: Data object for solvent or None
    :param device: torch device
    :param z_mask: atomic number to use for masking (default H=1)
    :param explain_solvent: whether to compute solvent atom importance
    :return: base prediction tensor shape [2], chromo importance tensor shape [num_atoms_chromo, 2],
             solvent importance tensor shape [num_atoms_solvent, 2] or None
    """
    model.eval()

    chromo_b = Batch.from_data_list([chromo_data]).to(device)
    solvent_b = Batch.from_data_list([solvent_data]).to(device) if solvent_data is not None else None

    base_pred = model(chromo_b, solvent_b).squeeze(0).detach().cpu()

    chromo_imp = []
    for i in range(chromo_data.num_nodes):
        chromo_i = _mask_atom_schnet(chromo_data.to(device), i, z_mask=z_mask)
        pred_i = model(Batch.from_data_list([chromo_i]).to(device), solvent_b).squeeze(0).detach().cpu()
        chromo_imp.append(base_pred - pred_i)
    chromo_imp = torch.stack(chromo_imp, dim=0)

    solvent_imp = None
    if solvent_data is not None and explain_solvent:
        solvent_list = []
        for j in range(solvent_data.num_nodes):
            solvent_j = _mask_atom_schnet(solvent_data.to(device), j, z_mask=z_mask)
            pred_j = model(chromo_b, Batch.from_data_list([solvent_j]).to(device)).squeeze(0).detach().cpu()
            solvent_list.append(base_pred - pred_j)
        solvent_imp = torch.stack(solvent_list, dim=0)

    return base_pred, chromo_imp, solvent_imp


def _remove_node_pyg(data: Data, node_idx: int) -> Data:
    """
    Return a new Data object with node `node_idx` removed.
    Works for Data with attributes: x, edge_index, optional edge_attr, optional pos.

    :param data: PyG Data object
    :param node_idx: index of node to remove
    :return: new Data object with the node removed
    """
    n = data.num_nodes
    if n <= 1:
        # can't remove the only node; return as-is
        return data

    keep = torch.ones(n, dtype=torch.bool, device=data.x.device)
    keep[node_idx] = False

    # map old node indices -> new node indices
    new_index = -torch.ones(n, dtype=torch.long, device=data.x.device)
    new_index[keep] = torch.arange(keep.sum(), device=data.x.device)

    # filter node features
    x_new = data.x[keep]
    pos_new = data.pos[keep] if hasattr(data, "pos") and data.pos is not None else None

    # filter edges: keep only edges where both endpoints are kept
    ei = data.edge_index
    src, dst = ei[0], ei[1]
    edge_keep = keep[src] & keep[dst]

    ei_new = ei[:, edge_keep]
    ei_new = new_index[ei_new]  # remap to [0..n_new-1]

    # filter edge_attr if present
    if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.numel() > 0:
        ea_new = data.edge_attr[edge_keep]
    else:
        ea_new = None

    out = Data(x=x_new, edge_index=ei_new)
    if ea_new is not None:
        out.edge_attr = ea_new
    if pos_new is not None:
        out.pos = pos_new
    return out


@torch.no_grad()
def atom_removal_attribution(
    model,
    chromo_data: Data,
    solvent_data: Data | None,
    device: torch.device,
    explain_solvent: bool = True,
):
    """
    Compute atom importance by removing each atom and measuring change in prediction.

    :param model: GNN model
    :param chromo_data: Data object for chromophore
    :param solvent_data: Data object for solvent or None
    :param device: torch device
    :param explain_solvent: whether to compute solvent atom importance
    :return: base prediction tensor shape [2], chromo importance tensor shape [num_atoms_chromo, 2],
             solvent importance tensor shape [num_atoms_solvent, 2] or None
    """
    model.eval()

    # Batch baseline
    chromo_b = Batch.from_data_list([chromo_data]).to(device)
    solvent_b = Batch.from_data_list([solvent_data]).to(device) if solvent_data is not None else None

    base_pred = model(chromo_b, solvent_b).squeeze(0).detach().cpu()  # [2]

    # Chromophore atom removal
    chromo_importance = []
    for i in range(chromo_data.num_nodes):
        chromo_i = _remove_node_pyg(chromo_data.to(device), i)
        chromo_i_b = Batch.from_data_list([chromo_i]).to(device)

        # solvent unchanged
        pred_i = model(chromo_i_b, solvent_b).squeeze(0).detach().cpu()
        chromo_importance.append(base_pred - pred_i)

    chromo_importance = torch.stack(chromo_importance, dim=0)  # [Nc, 2]

    # Solvent atom removal
    solvent_importance = None
    if solvent_data is not None and explain_solvent:
        solvent_importance_list = []
        for j in range(solvent_data.num_nodes):
            solvent_j = _remove_node_pyg(solvent_data.to(device), j)
            solvent_j_b = Batch.from_data_list([solvent_j]).to(device)

            pred_j = model(chromo_b, solvent_j_b).squeeze(0).detach().cpu()
            solvent_importance_list.append(base_pred - pred_j)

        solvent_importance = torch.stack(solvent_importance_list, dim=0)  # [Ns, 2]

    return base_pred, chromo_importance, solvent_importance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GNN for chromophore property prediction')
    parser.add_argument('--data-dir', '-d', required=True, help='Directory containing train.csv, val.csv, test.csv')
    parser.add_argument('--method', '-m', choices=['atom', 'atom_mask', 'bond'], default='atom', help='Attribution method: atom, atom_mask, or bond')

    args = parser.parse_args()

    test_path = os.path.join(args.data_dir, 'test.csv')

    # check if files exist
    for path in [test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    # load datasets
    print("Loading datasets...")
    test_dataset = ChromophoreDataset(test_path, position=True)

    # a sample from the dataset:
    num = 24
    chromo_data, solvent_data, desc, y = test_dataset[num]

    def load_trained_model(checkpoint_path: str, device: torch.device):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        args = checkpoint["args"]

        model = DualGNN(
            hidden_dim=args["hidden_dim"],
            use_solvent=not args["no_solvent"],
            gnn_type=args["gnn_type"],
            num_layers=args["num_layers"],
        ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return model, args

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_args = load_trained_model(
        "./models/scaffold_schnet/best_model.pt",
        device
    )

    model.to(device)

    if args.method == "atom":
        base_pred, chromo_imp, solvent_imp = atom_removal_attribution(
            model, chromo_data, solvent_data, device, explain_solvent=True
        )
    elif args.method == "atom_mask":
        base_pred, chromo_imp, solvent_imp = atom_mask_attribution_schnet(
            model, chromo_data, solvent_data, device, z_mask=1, explain_solvent=True
        )
    else:
        base_pred, bond_imp = bond_removal_attribution(
            model, chromo_data, solvent_data, device
        )

    # visualize absorption (index 0) for chromophore
    draw_hydrogens = True
    if not args.method == "bond":
        if draw_hydrogens:
            png_bytes = draw_mol_importance(
                smiles=test_dataset.df.iloc[test_dataset.valid_indices[num]]["Chromophore"],
                atom_importance=chromo_imp[:, 0].tolist(),
                title="Chromo atom importance (Absorption)"
            )
        else:
            smiles = test_dataset.df.iloc[test_dataset.valid_indices[num]]["Chromophore"]
            mol_noHs, heavy_imp = collapse_H_to_heavy(smiles, chromo_imp[:, 0].tolist())
            # draw using no-H SMILES (same smiles), but with add_hs=False and importance length = heavy atoms
            png_bytes = draw_mol_importance(
                smiles=smiles,
                atom_importance=heavy_imp,
                title="Chromo atom importance (Absorption, heavy atoms)",
                add_hs=False
            )

    smiles = test_dataset.df.iloc[test_dataset.valid_indices[num]]["Chromophore"]

    if args.method == "bond":
        png = draw_bond_importance(smiles, bond_imp)
        with open(f"bond_importance_abs_{num}.png", "wb") as f:
            f.write(png)
    else:
        with open(f"chromo_abs_importance_{num}.png", "wb") as f:
            f.write(png_bytes)
