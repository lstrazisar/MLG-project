import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import os

# Command line example:
# python3 src/split_dataset.py --input ./data/clean_chromophore_data.csv --output-dir ./data/splits/split_70_15_15 --train 70 --val 15 --test 15

# Parse command line arguments
parser = argparse.ArgumentParser(description='Split dataset by chromophore groups')
parser.add_argument('--input', '-i', required=True, help='Input CSV file')
parser.add_argument('--train', '-t', type=float, default=70.0, help='Train percentage (default: 70.0)')
parser.add_argument('--val', '-v', type=float, default=15.0, help='Validation percentage (default: 15.0)')
parser.add_argument('--test', '-s', type=float, default=15.0, help='Test percentage (default: 15.0)')
parser.add_argument('--output-dir', '-o', default='.', help='Output directory for split files (default: current directory)')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
parser.add_argument('--type', choices=['random', 'chromophore-grouped'], default='chromophore-grouped', help='Type of split: random or chromophore-grouped (default: chromophore-grouped)')

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

if args.type == 'chromophore-grouped':
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