import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import os
from collections import defaultdict

# Command line example:
# python3 src/split_dataset.py --input ./data/clean_chromophore_data.csv --output-dir ./data/splits/split_70_15_15 --train 70 --val 15 --test 15 --type scaffold

# Parse command line arguments
parser = argparse.ArgumentParser(description='Split dataset by chromophore groups or scaffolds')
parser.add_argument('--input', '-i', required=True, help='Input CSV file')
parser.add_argument('--train', '-t', type=float, default=80.0, help='Train percentage (default: 80.0)')
parser.add_argument('--val', '-v', type=float, default=10.0, help='Validation percentage (default: 10.0)')
parser.add_argument('--test', '-s', type=float, default=10.0, help='Test percentage (default: 10.0)')
parser.add_argument('--output-dir', '-o', default='.', help='Output directory for split files (default: current directory)')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
parser.add_argument('--type', choices=['random', 'chromophore-grouped', 'scaffold'], default='chromophore-grouped', 
                    help='Type of split: random, chromophore-grouped, or scaffold (default: chromophore-grouped)')
parser.add_argument('--smiles-column', default='SMILES', help='Name of the SMILES column (default: SMILES)')

args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Validate percentages sum to 100
total = args.train + args.val + args.test
if abs(total - 100.0) > 0.01:
    parser.error(f"Percentages must sum to 100, got {total}")

# Convert percentages to proportions
train_prop = args.train / 100.0
val_prop = args.val / 100.0
test_prop = args.test / 100.0

# Read the CSV file
df = pd.read_csv(args.input)

# filter out rows that do not have xyz optimization
import os
with_xyz = os.listdir('./data/xyz/chromophores/')
for i, chromophore in enumerate(with_xyz):
    chromophore = chromophore.split('.')[0].replace('&', '/').replace('$', '\\')
    with_xyz[i] = chromophore
    
df = df[df['Chromophore'].isin(with_xyz)]


def generate_scaffold(smiles):
    """Generate Bemis-Murcko scaffold for a molecule."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold
    except Exception as e:
        print(f"Warning: Could not generate scaffold for {smiles}: {e}")
        return None


def scaffold_split(df, smiles_column, train_prop, val_prop, test_prop, seed):
    """
    Split dataset by molecular scaffolds.
    Ensures molecules with the same scaffold stay in the same split.
    """
    np.random.seed(seed)
    
    # Generate scaffolds for all molecules
    print("Generating scaffolds...")
    scaffolds = defaultdict(list)
    invalid_count = 0
    
    for idx, row in df.iterrows():
        smiles = row[smiles_column]
        scaffold = generate_scaffold(smiles)
        if scaffold is not None:
            scaffolds[scaffold].append(idx)
        else:
            invalid_count += 1
    
    if invalid_count > 0:
        print(f"Warning: {invalid_count} molecules had invalid SMILES or could not generate scaffolds")
    
    # Sort scaffolds by size (number of molecules) in descending order
    # scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    scaffold_sets = list(scaffolds.values())
    
    # Calculate target sizes
    n_total = sum(len(s) for s in scaffold_sets)
    train_size = int(train_prop * n_total)
    val_size = int(val_prop * n_total)
    test_size = n_total - train_size - val_size
    
    # Assign scaffolds to splits
    train_indices = []
    val_indices = []
    test_indices = []
    
    train_count = 0
    val_count = 0
    test_count = 0
    
    for scaffold_set in scaffold_sets:
        # choose random split for this scaffold set
        r = np.random.rand()
        if r < 1/3 and test_count + len(scaffold_set) <= test_size * 1.02:
            test_indices.extend(scaffold_set)
            test_count += len(scaffold_set)
        elif r < 2/3 and val_count + len(scaffold_set) <= val_size * 1.02:
            val_indices.extend(scaffold_set)
            val_count += len(scaffold_set)
        else:
            train_indices.extend(scaffold_set)
            train_count += len(scaffold_set)
        
    
    # Create split dataframes
    train_df = df.loc[train_indices].copy()
    val_df = df.loc[val_indices].copy()
    test_df = df.loc[test_indices].copy()
    
    # Print statistics
    print(f"\nScaffold split statistics:")
    print(f"Total samples: {n_total}")
    print(f"Unique scaffolds: {len(scaffolds)}")
    print(f"\nTrain set:")
    print(f"  Samples: {len(train_df)} ({len(train_df)/n_total*100:.1f}%)")
    print(f"  Unique scaffolds: {len(set(generate_scaffold(df.loc[i, smiles_column]) for i in train_indices))}")
    print(f"\nValidation set:")
    print(f"  Samples: {len(val_df)} ({len(val_df)/n_total*100:.1f}%)")
    print(f"  Unique scaffolds: {len(set(generate_scaffold(df.loc[i, smiles_column]) for i in val_indices))}")
    print(f"\nTest set:")
    print(f"  Samples: {len(test_df)} ({len(test_df)/n_total*100:.1f}%)")
    print(f"  Unique scaffolds: {len(set(generate_scaffold(df.loc[i, smiles_column]) for i in test_indices))}")
    
    return train_df, val_df, test_df


if args.type == 'scaffold':
    # Check if RDKit is available
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        parser.error("RDKit is required for scaffold splitting. Install with: pip install rdkit")
    
    # Check if SMILES column exists
    if args.smiles_column not in df.columns:
        parser.error(f"SMILES column '{args.smiles_column}' not found in dataset. Available columns: {list(df.columns)}")
    
    # Perform scaffold split
    train_df, val_df, test_df = scaffold_split(
        df, args.smiles_column, train_prop, val_prop, test_prop, args.seed
    )
    
    # Save the splits
    train_path = os.path.join(args.output_dir, 'train.csv')
    val_path = os.path.join(args.output_dir, 'val.csv')
    test_path = os.path.join(args.output_dir, 'test.csv')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSplits saved to {args.output_dir}/")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print(f"  - {test_path}")

elif args.type == 'chromophore-grouped':
    # Get unique chromophores
    unique_chromophores = df['Chromophore'].unique()

    # Split chromophores into train, val, test
    train_chromophores, temp_chromophores = train_test_split(
        unique_chromophores, 
        test_size=(1 - train_prop), 
        random_state=args.seed
    )

    # Calculate proportion of val in the remaining data
    val_prop_adjusted = val_prop / (val_prop + test_prop)
    val_chromophores, test_chromophores = train_test_split(
        temp_chromophores, 
        test_size=(1 - val_prop_adjusted), 
        random_state=args.seed
    )

    # Create splits based on chromophore groups
    train_df = df[df['Chromophore'].isin(train_chromophores)]
    val_df = df[df['Chromophore'].isin(val_chromophores)]
    test_df = df[df['Chromophore'].isin(test_chromophores)]

    # Print split statistics
    print(f"Total samples: {len(df)}")
    print(f"Unique chromophores: {len(unique_chromophores)}")
    print(f"\nTrain set:")
    print(f"  Samples: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Unique chromophores: {len(train_chromophores)}")
    print(f"\nValidation set:")
    print(f"  Samples: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Unique chromophores: {len(val_chromophores)}")
    print(f"\nTest set:")
    print(f"  Samples: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Unique chromophores: {len(test_chromophores)}")

    # Save the splits to separate CSV files
    train_path = os.path.join(args.output_dir, 'train.csv')
    val_path = os.path.join(args.output_dir, 'val.csv')
    test_path = os.path.join(args.output_dir, 'test.csv')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSplits saved to {args.output_dir}/")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print(f"  - {test_path}")
    
elif args.type == 'random':
    # Randomly shuffle and split the dataset
    train_df, temp_df = train_test_split(
        df, 
        test_size=(1 - train_prop), 
        random_state=args.seed
    )

    val_prop_adjusted = val_prop / (val_prop + test_prop)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=(1 - val_prop_adjusted), 
        random_state=args.seed
    )

    # Print split statistics
    print(f"Total samples: {len(df)}")
    print(f"\nTrain set:")
    print(f"  Samples: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"\nValidation set:")
    print(f"  Samples: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"\nTest set:")
    print(f"  Samples: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    # Save the splits to separate CSV files
    train_path = os.path.join(args.output_dir, 'train.csv')
    val_path = os.path.join(args.output_dir, 'val.csv')
    test_path = os.path.join(args.output_dir, 'test.csv')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSplits saved to {args.output_dir}/")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print(f"  - {test_path}")