#!/usr/bin/env python3
"""
Diabetes Dataset Preprocessing Script
This script performs data preprocessing based on the analysis recommendations
to prepare the data for optimal model training performance.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import joblib

def find_dataset_file():
    """Find the diabetes dataset file in the project directories."""
    search_paths = [
        '/mnt/data/',
        '/mnt/imported/data/',
        '/mnt/code/',
        './',
        '../'
    ]
    
    possible_names = [
        'diabetes_dataset.csv',
        'Diabetes Dataset.csv',
        'diabetes.csv'
    ]
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if any(name.lower() in file.lower() for name in ['diabetes', 'dataset']) and file.endswith('.csv'):
                        full_path = os.path.join(root, file)
                        return full_path
    
    # Also check current directory specifically
    for name in possible_names:
        if os.path.exists(name):
            return name
    
    return None

def handle_outliers(df, feature_cols, method='clip', z_threshold=3):
    """
    Handle outliers in the dataset using specified method.
    
    Args:
        df: DataFrame with the data
        feature_cols: List of feature columns to process
        method: 'clip', 'remove', or 'cap' (default: 'clip')
        z_threshold: Z-score threshold for outlier detection (default: 3)
    
    Returns:
        DataFrame with outliers handled
    """
    df_processed = df.copy()
    outlier_info = {}
    
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64']:
            # Calculate IQR-based outlier bounds
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers before processing
            outliers_before = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
            
            if method == 'clip':
                # Clip outliers to the bounds
                df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
            elif method == 'cap':
                # Cap outliers to percentile values
                df_processed[col] = df_processed[col].clip(
                    lower=df[col].quantile(0.05), 
                    upper=df[col].quantile(0.95)
                )
            elif method == 'remove':
                # Mark rows with outliers for removal (handled later)
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                df_processed = df_processed[~outlier_mask]
            
            # Count outliers after processing
            outliers_after = len(df_processed[(df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)])
            
            outlier_info[col] = {
                'before': outliers_before,
                'after': outliers_after,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
    
    return df_processed, outlier_info

def preprocess_diabetes_data(input_file, output_file='diabetes_dataset_preprocessed.csv', outlier_method='clip'):
    """
    Main preprocessing function for the diabetes dataset.
    
    Args:
        input_file: Path to the original dataset
        output_file: Path to save the preprocessed dataset
        outlier_method: Method for handling outliers ('clip', 'cap', 'remove')
    
    Returns:
        Dictionary with preprocessing information
    """
    
    print("=" * 80)
    print("DIABETES DATASET PREPROCESSING")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Outlier handling method: {outlier_method}")
    print()
    
    # Load the dataset
    try:
        df = pd.read_csv(input_file)
        print(f"✓ Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None
    
    # Identify feature columns (excluding target)
    feature_cols = [col for col in df.columns if col != 'is_diabetic']
    
    print(f"Feature columns: {feature_cols}")
    print(f"Target column: is_diabetic")
    print()
    
    # 1. HANDLE OUTLIERS
    print("1. HANDLING OUTLIERS")
    print("-" * 40)
    
    df_processed, outlier_info = handle_outliers(df, feature_cols, method=outlier_method)
    
    for col, info in outlier_info.items():
        reduction = info['before'] - info['after']
        reduction_pct = (reduction / info['before'] * 100) if info['before'] > 0 else 0
        print(f"{col}:")
        print(f"  Outliers before: {info['before']}")
        print(f"  Outliers after: {info['after']}")
        print(f"  Reduction: {reduction} ({reduction_pct:.1f}%)")
        print(f"  Bounds: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")
        print()
    
    # 2. FEATURE SCALING
    print("2. FEATURE SCALING")
    print("-" * 40)
    
    # Use RobustScaler as it's less sensitive to outliers
    scaler = RobustScaler()
    
    # Separate features and target
    X = df_processed[feature_cols]
    y = df_processed['is_diabetic']
    
    # Fit and transform features
    X_scaled = scaler.fit_transform(X)
    
    # Create DataFrame with scaled features
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df_processed.index)
    
    # Combine scaled features with target
    df_final = pd.concat([X_scaled_df, y], axis=1)
    
    print("✓ Applied RobustScaler to all features")
    print("  RobustScaler uses median and IQR, making it robust to outliers")
    print()
    
    # 3. FEATURE STATISTICS COMPARISON
    print("3. BEFORE/AFTER FEATURE STATISTICS")
    print("-" * 40)
    
    print("Original feature statistics:")
    print(df[feature_cols].describe().round(3))
    print()
    
    print("Scaled feature statistics:")
    print(X_scaled_df.describe().round(3))
    print()
    
    # 4. SAVE PREPROCESSED DATA
    print("4. SAVING PREPROCESSED DATA")
    print("-" * 40)
    
    try:
        df_final.to_csv(output_file, index=False)
        print(f"✓ Preprocessed dataset saved to: {output_file}")
        print(f"  Shape: {df_final.shape}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Samples: {len(df_final)}")
    except Exception as e:
        print(f"✗ Error saving preprocessed dataset: {e}")
        return None
    
    # 5. SAVE SCALER FOR FUTURE USE
    scaler_file = 'diabetes_scaler.joblib'
    try:
        joblib.dump(scaler, scaler_file)
        print(f"✓ Scaler saved to: {scaler_file}")
        print("  This can be used to transform new data for prediction")
    except Exception as e:
        print(f"⚠️  Warning: Could not save scaler: {e}")
    
    print()
    
    # 6. CREATE TRAIN/VALIDATION SPLIT FILES
    print("5. CREATING TRAIN/VALIDATION SPLITS")
    print("-" * 40)
    
    # Create train/validation split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled_df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save training set
    train_df = pd.concat([X_train, y_train], axis=1)
    train_file = 'diabetes_dataset_train.csv'
    train_df.to_csv(train_file, index=False)
    
    # Save validation set
    val_df = pd.concat([X_val, y_val], axis=1)
    val_file = 'diabetes_dataset_val.csv'
    val_df.to_csv(val_file, index=False)
    
    print(f"✓ Training set saved to: {train_file}")
    print(f"  Shape: {train_df.shape}")
    print(f"  Class distribution: {y_train.value_counts().to_dict()}")
    print()
    
    print(f"✓ Validation set saved to: {val_file}")
    print(f"  Shape: {val_df.shape}")
    print(f"  Class distribution: {y_val.value_counts().to_dict()}")
    print()
    
    # 7. SUMMARY AND RECOMMENDATIONS
    print("6. PREPROCESSING SUMMARY")
    print("-" * 40)
    
    original_samples = len(df)
    final_samples = len(df_final)
    sample_retention = (final_samples / original_samples) * 100
    
    print(f"Original samples: {original_samples:,}")
    print(f"Final samples: {final_samples:,}")
    print(f"Sample retention: {sample_retention:.1f}%")
    print()
    
    print("Preprocessing steps applied:")
    print(f"  1. Outlier handling: {outlier_method}")
    print("  2. Feature scaling: RobustScaler")
    print("  3. Train/validation split: 80/20")
    print()
    
    print("Files generated:")
    print(f"  • {output_file} - Complete preprocessed dataset")
    print(f"  • {train_file} - Training set (ready for model training)")
    print(f"  • {val_file} - Validation set (ready for model evaluation)")
    print(f"  • {scaler_file} - Fitted scaler (for new data preprocessing)")
    print()
    
    print("Next steps for model training:")
    print("  1. Use the training set for model training")
    print("  2. Use the validation set for model evaluation")
    print("  3. Apply the saved scaler to any new data before prediction")
    print("  4. Consider the excellent feature correlations:")
    print("     - weight: 0.885 (strongest predictor)")
    print("     - calories_wk: 0.790")
    print("     - annual_income: 0.745")
    print("     - hrs_exercise_wk: 0.607")
    
    print()
    print("=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    
    # Return summary information
    return {
        'original_shape': df.shape,
        'final_shape': df_final.shape,
        'sample_retention': sample_retention,
        'outlier_info': outlier_info,
        'files_created': [output_file, train_file, val_file, scaler_file],
        'feature_columns': feature_cols,
        'scaling_method': 'RobustScaler'
    }

def main():
    """Main function to run the preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess diabetes dataset for model training")
    parser.add_argument("--outlier_method", choices=['clip', 'cap', 'remove'], 
                       default='clip', help="Method for handling outliers")
    parser.add_argument("--output_file", default='diabetes_dataset_preprocessed.csv',
                       help="Output file name for preprocessed data")
    
    args = parser.parse_args()
    
    print("Starting diabetes dataset preprocessing...")
    print()
    
    # Find the dataset file
    dataset_path = find_dataset_file()
    
    if dataset_path is None:
        print("✗ Could not find diabetes dataset file!")
        print("Please ensure the dataset is available in one of the expected locations.")
        sys.exit(1)
    
    # Preprocess the dataset
    result = preprocess_diabetes_data(
        input_file=dataset_path,
        output_file=args.output_file,
        outlier_method=args.outlier_method
    )
    
    if result is not None:
        print("Preprocessing completed successfully!")
        
        # Save preprocessing summary
        with open('diabetes_preprocessing_summary.txt', 'w') as f:
            f.write("Diabetes Dataset Preprocessing Summary\n")
            f.write("======================================\n")
            f.write(f"Original shape: {result['original_shape']}\n")
            f.write(f"Final shape: {result['final_shape']}\n")
            f.write(f"Sample retention: {result['sample_retention']:.1f}%\n")
            f.write(f"Scaling method: {result['scaling_method']}\n")
            f.write(f"Feature columns: {result['feature_columns']}\n")
            f.write(f"Files created: {result['files_created']}\n")
        
        print("Summary saved to: diabetes_preprocessing_summary.txt")
    else:
        print("Preprocessing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
