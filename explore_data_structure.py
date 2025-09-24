#!/usr/bin/env python3
"""
Script to explore the data structure and locate the diabetes dataset.
This script will recursively list all files in the data directories.
"""

import os
import pandas as pd

def explore_directory_structure():
    """Explore the data directory structure"""
    data_paths = ['/mnt/data/', '/mnt/imported/data/']
    
    print("=== EXPLORING DATA DIRECTORY STRUCTURE ===\n")
    
    for data_path in data_paths:
        print(f"Exploring: {data_path}")
        if os.path.exists(data_path):
            print(f"✓ Directory exists: {data_path}")
            
            # Recursively list all files
            for root, dirs, files in os.walk(data_path):
                level = root.replace(data_path, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                
                sub_indent = ' ' * 2 * (level + 1)
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    print(f"{sub_indent}{file} ({file_size} bytes)")
        else:
            print(f"✗ Directory does not exist: {data_path}")
        print()
    
    # Also check current directory for any datasets
    print("Checking current directory for datasets...")
    current_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if current_files:
        print(f"Found CSV files in current directory: {current_files}")
    else:
        print("No CSV files found in current directory")

def find_diabetes_datasets():
    """Find potential diabetes datasets"""
    print("\n=== SEARCHING FOR DIABETES DATASETS ===\n")
    
    search_paths = ['/mnt/data/', '/mnt/imported/data/', '.']
    diabetes_files = []
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if 'diabetes' in file.lower() or file.endswith('.csv'):
                        file_path = os.path.join(root, file)
                        diabetes_files.append(file_path)
                        print(f"Found potential dataset: {file_path}")
    
    return diabetes_files

def analyze_dataset_preview(file_path):
    """Analyze and preview a dataset"""
    print(f"\n=== ANALYZING DATASET: {file_path} ===\n")
    
    try:
        # Read the dataset
        df = pd.read_csv(file_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nDataset info:")
        print(df.info())
        print(f"\nBasic statistics:")
        print(df.describe())
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            print(f"\nMissing values:")
            print(missing_values[missing_values > 0])
        else:
            print("\nNo missing values found")
            
        # Check target variable if it exists
        potential_targets = ['is_diabetic', 'diabetes', 'target', 'label', 'outcome']
        target_col = None
        for col in potential_targets:
            if col in df.columns:
                target_col = col
                break
        
        if target_col:
            print(f"\nTarget variable '{target_col}' distribution:")
            print(df[target_col].value_counts())
            print(f"Target variable type: {df[target_col].dtype}")
        else:
            print("\nNo obvious target variable found")
            
    except Exception as e:
        print(f"Error analyzing dataset: {e}")

if __name__ == "__main__":
    # Explore directory structure
    explore_directory_structure()
    
    # Find potential diabetes datasets
    datasets = find_diabetes_datasets()
    
    # Analyze each found dataset
    for dataset in datasets:
        if os.path.exists(dataset):
            analyze_dataset_preview(dataset)
        else:
            print(f"Dataset file does not exist: {dataset}")
