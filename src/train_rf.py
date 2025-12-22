import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
import os
from tqdm import tqdm

# Command line example:
# python3 train_rf.py --input-dir ./data/splits/scaffold_70_15_15 --target "Absorption max (nm)"

parser = argparse.ArgumentParser(description='Train Random Forest with RDKit descriptors')
parser.add_argument('--input-dir', '-i', required=True, help='Directory containing train.csv, val.csv, test.csv')
parser.add_argument('--target', '-t', required=True, help='Target column name (e.g., "Absorption max (nm)")')
parser.add_argument('--n-estimators', type=int, default=100, help='Number of trees (default: 100)')
parser.add_argument('--max-depth', type=int, default=None, help='Max depth of trees (default: None)')
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')

args = parser.parse_args()

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

# Load data
train_path = os.path.join(args.input_dir, 'train.csv')
val_path = os.path.join(args.input_dir, 'val.csv')
test_path = os.path.join(args.input_dir, 'test.csv')

print("Loading datasets...")
train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

print(f"Train: {len(train_df)} samples")
print(f"Val: {len(val_df)} samples")
print(f"Test: {len(test_df)} samples")

# Check if target column exists
if args.target not in train_df.columns:
    raise ValueError(f"Target column '{args.target}' not found. Available: {list(train_df.columns)}")

# Featurize molecules
print("\n=== Featurizing Training Set ===")
X_train = featurize_dataframe(train_df)
y_train = train_df.loc[X_train.index, args.target]

# Get valid columns from training set
valid_train_features = X_train.columns.tolist()

print("\n=== Featurizing Validation Set ===")
X_val = featurize_dataframe(val_df, valid_columns=valid_train_features)
y_val = val_df.loc[X_val.index, args.target]

print("\n=== Featurizing Test Set ===")
X_test = featurize_dataframe(test_df, valid_columns=valid_train_features)
y_test = test_df.loc[X_test.index, args.target]

# Convert to float64 to avoid precision issues
X_train = X_train.astype(np.float64)
X_val = X_val.astype(np.float64)
X_test = X_test.astype(np.float64)

print(f"\nFinal shapes:")
print(f"  Train: {X_train.shape}")
print(f"  Val:   {X_val.shape}")
print(f"  Test:  {X_test.shape}")

# Train Random Forest
print(f"\n=== Training Random Forest ===")
print(f"n_estimators: {args.n_estimators}")
print(f"max_depth: {args.max_depth}")
print(f"random_state: {args.seed}")

rf = RandomForestRegressor(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    random_state=args.seed,
    n_jobs=-1,
    verbose=1
)

print("="*50)
print(type(np.isnan(X_train).sum()))
print("NaNs:", np.isnan(X_train).sum())
# print max value in X_train and the column name
max_value = np.max(X_train.values)
max_value_column = X_train.columns[np.argmax(np.max(X_train.values, axis=0))]
print(f"Max value in X_train: {max_value} (Column: {max_value_column})")


rf.fit(X_train, y_train)

# Make predictions
print("\n=== Making Predictions ===")
train_pred = rf.predict(X_train)
val_pred = rf.predict(X_val)
test_pred = rf.predict(X_test)

# Evaluate
def evaluate(y_true, y_pred, set_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{set_name} Set:")
    print(f"  MAE:  {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R²:   {r2:.3f}")
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

print("\n" + "="*50)
print("RESULTS")
print("="*50)

train_metrics = evaluate(y_train, train_pred, "Train")
val_metrics = evaluate(y_val, val_pred, "Validation")
test_metrics = evaluate(y_test, test_pred, "Test")

# Feature importance
print("\n=== Top 10 Most Important Features ===")
feature_importance = pd.DataFrame({
    'feature': valid_train_features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# Save results
results = {
    'target': args.target,
    'n_estimators': args.n_estimators,
    'max_depth': args.max_depth,
    'n_features': len(valid_train_features),
    'train_samples': len(y_train),
    'val_samples': len(y_val),
    'test_samples': len(y_test),
    **{f'train_{k}': v for k, v in train_metrics.items()},
    **{f'val_{k}': v for k, v in val_metrics.items()},
    **{f'test_{k}': v for k, v in test_metrics.items()}
}

results_df = pd.DataFrame([results])
results_path = os.path.join(args.input_dir, 'rf_results.csv')
results_df.to_csv(results_path, index=False)
print(f"\nResults saved to {results_path}")

# Save predictions
predictions_df = pd.DataFrame({
    'true': y_test,
    'predicted': test_pred,
    'error': y_test - test_pred
})
predictions_path = os.path.join(args.input_dir, 'test_predictions.csv')
predictions_df.to_csv(predictions_path, index=False)
print(f"Test predictions saved to {predictions_path}")