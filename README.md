# Chromophore Property Prediction with Graph Neural Networks

This repository contains the code for predicting chromophore absorption and emission wavelengths using Graph Neural Networks (GNNs), developed as part of the project for the course Machine Learning with Graphs. The models process molecular structures (SMILES) of chromophores and solvents to predict optical properties.

## Installation

We recommend running the code on Linux. The necessary dependencies can be installed via:

```bash
pip install rdkit pandas numpy scikit-learn tqdm torch==2.8.0 torch-geometric
pip install --only-binary=:all:  torch-cluster  \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

## Features

- **Multiple GNN Architectures**: Support for GCN, GAT, GIN and SchNet
- **Dual-molecule processing**: Separate encoding for chromophore and solvent
- **Flexible architecture**: Configurable hidden dimensions, number of layers, and dropout
- **Comprehensive evaluation**: Reports MAE for both absorption and emission wavelengths
- **Data splitting**: Ensures all samples with the same chromophore stay in the same split (scaffold splitting)


## Dataset Format

Your CSV files should contain the following columns:
- `Tag`: Identifier for the sample
- `Chromophore`: SMILES string of the chromophore molecule
- `Solvent`: SMILES string of the solvent molecule
- `Absorption max (nm)`: Target absorption wavelength
- `Emission max (nm)`: Target emission wavelength

An example of such a dataset is our own, on which we trained the models - `data/clean_chromophore_data.csv`.

## Usage

### Step 1: Split the Dataset

Split your dataset into train/val/test sets while keeping all samples with the same chromophore together:

```bash
python3 ./src/split_dataset.py \
    --input data.csv \
    --output-dir ./splits/split_70_15_15 \
    --train 70 \
    --val 15 \
    --test 15 \
    --seed 42
```

**Arguments:**
- `--input, -i`: Input CSV file (required)
- `--output-dir, -o`: Output directory for split files (default: current directory)
- `--train, -t`: Train percentage (default: 70.0)
- `--val, -v`: Validation percentage (default: 15.0)
- `--test, -s`: Test percentage (default: 15.0)
- `--seed`: Random seed for reproducibility (default: 42)

This creates three files: `train.csv`, `val.csv`, and `test.csv` in the output directory.

Alternatively, use any of our own splits in `data/splits`.

### Step 2: Train the GNN Model

Train a GNN model on the split data. Example:

```bash
python3 ./src/train_gnn.py \
    --data-dir ./data/splits/scaffold_split_80_10_10 \
    --output-dir ./models \
    --gnn-type gin \
    --num-layers 3 \
    --hidden-dim 128 \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001
```

**Arguments:**
- `--data-dir, -d`: Directory containing train.csv, val.csv, test.csv (required)
- `--output-dir, -o`: Output directory for models (default: `./models`)
- `--gnn-type`: Type of GNN architecture - `gcn`, `gat`, `gin` or 'schnet' (default: gcn)
- `--num-layers`: Number of GNN layers (default: 3)
- `--hidden-dim`: Hidden dimension size (default: 128)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 0.001)
- `--no-solvent`: Use only chromophore, ignore solvent (for baseline)
- `--device`: Device to use - `cuda` or `cpu` (default: auto-detect)

### Step 3: Evaluate Results

Training automatically evaluates on the test set and saves:
- `best_model.pt`: Best model checkpoint based on validation loss
- `history.json`: Training and validation metrics per epoch
- `test_results.json`: Final test set performance

## GNN Architectures

### GCN (Graph Convolutional Network)

```bash
python train_gnn.py -d ./splits/scaffold_split_80_10_10 --gnn-type gcn --num-layers 2 --hidden-dim 64
```

### GAT (Graph Attention Network)

```bash
python train_gnn.py -d ./splits/scaffold_split_80_10_10 --gnn-type gat --num-layers 3 --hidden-dim 128
```

### GIN (Graph Isomorphism Network)

```bash
python train_gnn.py -d ./splits/scaffold_split_80_10_10 --gnn-type gin --num-layers 4 --hidden-dim 128
```

## Model Architecture Details

The `DualGNN` model:
1. Processes chromophore and solvent separately through independent GNN layers
2. Uses global pooling (mean for GCN/GAT, sum for GIN) to get graph-level embeddings
3. Concatenates chromophore and solvent embeddings
4. Passes through MLP to predict absorption and emission wavelengths

Each GNN layer includes:
- Graph convolution (GCN/GAT/GIN)
- Batch normalization
- ReLU activation
- Dropout (0.2)

## Output Interpretation

### During Training
```
Epoch 50/100
Training: 100%|████████| 350/350 [00:15<00:00]
Evaluating: 100%|████████| 47/47 [00:01<00:00]
Train Loss: 245.3421 | MAE - Absorption: 12.34 nm, Emission: 15.67 nm
Val Loss: 312.5643 | MAE - Absorption: 14.89 nm, Emission: 18.23 nm
✓ Saved best model
```

### Test Results (test_results.json)
```json
{
  "test_loss": 298.7654,
  "test_mae_absorption": 13.45,
  "test_mae_emission": 17.89
}
```


