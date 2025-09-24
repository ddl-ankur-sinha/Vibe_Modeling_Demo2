#!/usr/bin/env python3
"""
Comprehensive Diabetes Dataset Analysis Script
This script analyzes the diabetes dataset for model training suitability
and provides recommendations for data preprocessing.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

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
            print(f"Searching in: {search_path}")
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if any(name.lower() in file.lower() for name in ['diabetes', 'dataset']) and file.endswith('.csv'):
                        full_path = os.path.join(root, file)
                        print(f"Found potential dataset: {full_path}")
                        return full_path
    
    # Also check current directory specifically
    for name in possible_names:
        if os.path.exists(name):
            print(f"Found dataset in current directory: {name}")
            return name
    
    return None

def analyze_dataset(file_path):
    """Perform comprehensive analysis of the diabetes dataset."""
    
    print("=" * 80)
    print("DIABETES DATASET ANALYSIS REPORT")
    print("=" * 80)
    print(f"Dataset file: {file_path}")
    print()
    
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        print(f"✓ Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        print()
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None
    
    # 1. BASIC DATASET INFORMATION
    print("1. BASIC DATASET INFORMATION")
    print("-" * 40)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print()
    
    # 2. TARGET VARIABLE ANALYSIS
    print("2. TARGET VARIABLE ANALYSIS")
    print("-" * 40)
    if 'is_diabetic' in df.columns:
        target_counts = df['is_diabetic'].value_counts()
        target_pct = df['is_diabetic'].value_counts(normalize=True) * 100
        
        print("Target variable distribution:")
        print(f"Non-diabetic (0): {target_counts[0]:,} ({target_pct[0]:.1f}%)")
        print(f"Diabetic (1): {target_counts[1]:,} ({target_pct[1]:.1f}%)")
        
        # Check for class imbalance
        imbalance_ratio = target_counts.max() / target_counts.min()
        print(f"Class imbalance ratio: {imbalance_ratio:.2f}")
        if imbalance_ratio > 3:
            print("⚠️  WARNING: Significant class imbalance detected!")
        else:
            print("✓ Class distribution is reasonably balanced")
    else:
        print("✗ Target variable 'is_diabetic' not found!")
        return None
    print()
    
    # 3. FEATURE ANALYSIS
    print("3. FEATURE ANALYSIS")
    print("-" * 40)
    feature_cols = [col for col in df.columns if col != 'is_diabetic']
    
    print("Feature summary statistics:")
    print(df[feature_cols].describe())
    print()
    
    # 4. MISSING VALUES ANALYSIS
    print("4. MISSING VALUES ANALYSIS")
    print("-" * 40)
    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df)) * 100
    
    missing_summary = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_pct
    })
    missing_summary = missing_summary[missing_summary['Missing Count'] > 0]
    
    if len(missing_summary) == 0:
        print("✓ No missing values found in the dataset")
    else:
        print("Missing values detected:")
        print(missing_summary)
        print("⚠️  Missing values need to be handled before training")
    print()
    
    # 5. FEATURE DISTRIBUTIONS AND OUTLIERS
    print("5. FEATURE DISTRIBUTIONS AND OUTLIERS")
    print("-" * 40)
    
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64']:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outlier_pct = len(outliers) / len(df) * 100
            
            print(f"{col}:")
            print(f"  Range: {df[col].min():.2f} to {df[col].max():.2f}")
            print(f"  Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}")
            print(f"  Outliers: {len(outliers)} ({outlier_pct:.1f}%)")
            
            if outlier_pct > 5:
                print(f"  ⚠️  High percentage of outliers detected")
            print()
    
    # 6. FEATURE CORRELATIONS
    print("6. FEATURE CORRELATIONS WITH TARGET")
    print("-" * 40)
    correlations = df[feature_cols].corrwith(df['is_diabetic']).abs().sort_values(ascending=False)
    
    print("Absolute correlations with diabetes target:")
    for feature, corr in correlations.items():
        strength = "Strong" if corr > 0.5 else "Moderate" if corr > 0.3 else "Weak"
        print(f"  {feature}: {corr:.3f} ({strength})")
    print()
    
    # 7. DATA QUALITY ASSESSMENT
    print("7. DATA QUALITY ASSESSMENT")
    print("-" * 40)
    
    quality_issues = []
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        quality_issues.append(f"Duplicate rows: {duplicates}")
    
    # Check for constant features
    constant_features = [col for col in feature_cols if df[col].nunique() <= 1]
    if constant_features:
        quality_issues.append(f"Constant features: {constant_features}")
    
    # Check for highly correlated features
    feature_corr_matrix = df[feature_cols].corr().abs()
    high_corr_pairs = []
    for i in range(len(feature_corr_matrix.columns)):
        for j in range(i+1, len(feature_corr_matrix.columns)):
            if feature_corr_matrix.iloc[i, j] > 0.9:
                high_corr_pairs.append((feature_corr_matrix.columns[i], feature_corr_matrix.columns[j]))
    
    if high_corr_pairs:
        quality_issues.append(f"Highly correlated feature pairs: {high_corr_pairs}")
    
    if quality_issues:
        print("Data quality issues found:")
        for issue in quality_issues:
            print(f"  ⚠️  {issue}")
    else:
        print("✓ No major data quality issues detected")
    print()
    
    # 8. MODEL TRAINING SUITABILITY ASSESSMENT
    print("8. MODEL TRAINING SUITABILITY ASSESSMENT")
    print("-" * 40)
    
    suitability_score = 0
    max_score = 7
    
    # Dataset size
    if len(df) >= 10000:
        print("✓ Dataset size is adequate (≥10,000 samples)")
        suitability_score += 1
    elif len(df) >= 1000:
        print("⚠️  Dataset size is moderate (1,000-10,000 samples)")
        suitability_score += 0.5
    else:
        print("✗ Dataset size is small (<1,000 samples)")
    
    # Feature count
    if len(feature_cols) >= 5:
        print("✓ Good number of features available")
        suitability_score += 1
    else:
        print("⚠️  Limited number of features")
        suitability_score += 0.5
    
    # Missing values
    if missing_data.sum() == 0:
        print("✓ No missing values")
        suitability_score += 1
    else:
        print("⚠️  Missing values present (need handling)")
    
    # Class balance
    if imbalance_ratio <= 3:
        print("✓ Classes are reasonably balanced")
        suitability_score += 1
    else:
        print("⚠️  Class imbalance present (may need balancing)")
        suitability_score += 0.5
    
    # Feature-target correlations
    strong_features = sum(1 for corr in correlations if corr > 0.3)
    if strong_features >= 3:
        print(f"✓ {strong_features} features show strong correlation with target")
        suitability_score += 1
    elif strong_features >= 1:
        print(f"⚠️  {strong_features} feature(s) show strong correlation with target")
        suitability_score += 0.5
    else:
        print("✗ No features show strong correlation with target")
    
    # Data types
    if all(df[col].dtype in ['int64', 'float64'] for col in feature_cols):
        print("✓ All features are numeric")
        suitability_score += 1
    else:
        print("⚠️  Some features are non-numeric (may need encoding)")
        suitability_score += 0.5
    
    # Outliers
    high_outlier_features = sum(1 for col in feature_cols 
                               if df[col].dtype in ['int64', 'float64'] and 
                               len(df[(df[col] < df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))) | 
                                      (df[col] > df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))]) / len(df) > 0.05)
    
    if high_outlier_features == 0:
        print("✓ No features with excessive outliers")
        suitability_score += 1
    else:
        print(f"⚠️  {high_outlier_features} feature(s) have high outlier percentage")
        suitability_score += 0.5
    
    print()
    suitability_percentage = (suitability_score / max_score) * 100
    print(f"OVERALL SUITABILITY SCORE: {suitability_score:.1f}/{max_score} ({suitability_percentage:.1f}%)")
    
    if suitability_percentage >= 80:
        print("✓ EXCELLENT - Dataset is highly suitable for diabetes prediction modeling")
    elif suitability_percentage >= 60:
        print("⚠️  GOOD - Dataset is suitable with minor preprocessing needs")
    elif suitability_percentage >= 40:
        print("⚠️  FAIR - Dataset needs significant preprocessing")
    else:
        print("✗ POOR - Dataset has major limitations for modeling")
    
    print()
    
    # 9. RECOMMENDATIONS
    print("9. PREPROCESSING RECOMMENDATIONS")
    print("-" * 40)
    
    recommendations = []
    
    if missing_data.sum() > 0:
        recommendations.append("Handle missing values using imputation or removal")
    
    if duplicates > 0:
        recommendations.append("Remove duplicate rows")
    
    if imbalance_ratio > 3:
        recommendations.append("Consider class balancing techniques (SMOTE, undersampling, etc.)")
    
    if high_outlier_features > 0:
        recommendations.append("Consider outlier treatment (clipping, transformation, or removal)")
    
    if constant_features:
        recommendations.append(f"Remove constant features: {constant_features}")
    
    if high_corr_pairs:
        recommendations.append("Consider removing one feature from highly correlated pairs")
    
    # Feature scaling recommendation
    feature_ranges = {}
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64']:
            feature_ranges[col] = df[col].max() - df[col].min()
    
    max_range = max(feature_ranges.values()) if feature_ranges else 0
    min_range = min(feature_ranges.values()) if feature_ranges else 0
    
    if max_range > 0 and max_range / min_range > 100:
        recommendations.append("Apply feature scaling (StandardScaler or MinMaxScaler)")
    
    if recommendations:
        print("Recommended preprocessing steps:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("✓ No major preprocessing needed - dataset is ready for training!")
    
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    return df

def main():
    """Main function to run the dataset analysis."""
    print("Starting diabetes dataset analysis...")
    print()
    
    # Find the dataset file
    dataset_path = find_dataset_file()
    
    if dataset_path is None:
        print("✗ Could not find diabetes dataset file!")
        print("Please ensure the dataset is available in one of the expected locations.")
        sys.exit(1)
    
    # Analyze the dataset
    df = analyze_dataset(dataset_path)
    
    if df is not None:
        print("Analysis completed successfully!")
        
        # Save a summary to a text file
        with open('diabetes_dataset_analysis_summary.txt', 'w') as f:
            f.write(f"Dataset Analysis Summary\n")
            f.write(f"========================\n")
            f.write(f"Dataset path: {dataset_path}\n")
            f.write(f"Shape: {df.shape}\n")
            f.write(f"Columns: {list(df.columns)}\n")
            f.write(f"Target distribution:\n{df['is_diabetic'].value_counts()}\n")
            f.write(f"Feature statistics:\n{df.describe()}\n")
        
        print("Summary saved to: diabetes_dataset_analysis_summary.txt")
    else:
        print("Analysis failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
