#!/usr/bin/env python3
"""
Enhanced Diabetes Dataset Analysis for Model Training Evaluation
This script provides a comprehensive analysis of the diabetes dataset
to evaluate its suitability for machine learning and identify improvements.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

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

def detect_outliers_iqr(data, multiplier=1.5):
    """Detect outliers using IQR method."""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(data, threshold=3):
    """Detect outliers using Z-score method."""
    z_scores = np.abs(stats.zscore(data))
    outliers = z_scores > threshold
    return outliers

def analyze_feature_distributions(df, feature_cols):
    """Analyze the distribution of each feature."""
    print("\n" + "="*60)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    for col in feature_cols:
        print(f"\n--- {col.upper()} ---")
        
        # Basic statistics
        print(f"Mean: {df[col].mean():.4f}")
        print(f"Median: {df[col].median():.4f}")
        print(f"Std: {df[col].std():.4f}")
        print(f"Min: {df[col].min():.4f}")
        print(f"Max: {df[col].max():.4f}")
        
        # Skewness and kurtosis
        skewness = stats.skew(df[col])
        kurtosis = stats.kurtosis(df[col])
        print(f"Skewness: {skewness:.4f}")
        print(f"Kurtosis: {kurtosis:.4f}")
        
        # Normality test
        try:
            _, p_value = stats.normaltest(df[col])
            print(f"Normality test p-value: {p_value:.6f}")
            if p_value < 0.05:
                print("  → Distribution is NOT normal (p < 0.05)")
            else:
                print("  → Distribution appears normal (p >= 0.05)")
        except:
            print("  → Could not perform normality test")
        
        # Outlier detection
        outliers_iqr, lower, upper = detect_outliers_iqr(df[col])
        outliers_zscore = detect_outliers_zscore(df[col])
        
        print(f"Outliers (IQR method): {outliers_iqr.sum()} ({outliers_iqr.sum()/len(df)*100:.2f}%)")
        print(f"Outliers (Z-score method): {outliers_zscore.sum()} ({outliers_zscore.sum()/len(df)*100:.2f}%)")
        print(f"IQR bounds: [{lower:.2f}, {upper:.2f}]")

def analyze_class_imbalance(df, target_col):
    """Analyze class distribution and imbalance."""
    print("\n" + "="*60)
    print("CLASS IMBALANCE ANALYSIS")
    print("="*60)
    
    class_counts = df[target_col].value_counts()
    class_proportions = df[target_col].value_counts(normalize=True)
    
    print(f"\nClass distribution:")
    for class_val in class_counts.index:
        count = class_counts[class_val]
        prop = class_proportions[class_val]
        print(f"  Class {class_val}: {count} samples ({prop:.3f} or {prop*100:.1f}%)")
    
    # Calculate imbalance ratio
    majority_class = class_counts.max()
    minority_class = class_counts.min()
    imbalance_ratio = majority_class / minority_class
    
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 3:
        print("⚠️  SIGNIFICANT CLASS IMBALANCE DETECTED!")
        print("   Recommendations:")
        print("   - Consider SMOTE or other oversampling techniques")
        print("   - Use stratified sampling for train/validation split")
        print("   - Consider class weights in model training")
        print("   - Use appropriate evaluation metrics (F1, ROC-AUC, not just accuracy)")
    elif imbalance_ratio > 1.5:
        print("⚠️  Moderate class imbalance detected")
        print("   Recommendations:")
        print("   - Use stratified sampling")
        print("   - Monitor precision/recall for minority class")
    else:
        print("✅ Classes are relatively balanced")

def analyze_feature_correlations(df, feature_cols, target_col):
    """Analyze correlations between features and with target."""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Correlation with target
    print(f"\nFeature correlations with target ({target_col}):")
    target_corrs = df[feature_cols].corrwith(df[target_col]).sort_values(key=abs, ascending=False)
    
    for feature, corr in target_corrs.items():
        strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.3 else "Weak"
        direction = "positive" if corr > 0 else "negative"
        print(f"  {feature}: {corr:.4f} ({strength} {direction})")
    
    # Feature intercorrelations
    print(f"\nHigh feature intercorrelations (|r| > 0.7):")
    corr_matrix = df[feature_cols].corr()
    high_corr_pairs = []
    
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append((feature_cols[i], feature_cols[j], corr_val))
    
    if high_corr_pairs:
        for feat1, feat2, corr_val in high_corr_pairs:
            print(f"  {feat1} ↔ {feat2}: {corr_val:.4f}")
        print("\n⚠️  High correlations detected - consider feature selection or dimensionality reduction")
    else:
        print("  No high correlations detected (|r| > 0.7)")
        print("✅ Features appear to be relatively independent")

def analyze_missing_values(df):
    """Analyze missing values in the dataset."""
    print("\n" + "="*60)
    print("MISSING VALUES ANALYSIS")
    print("="*60)
    
    missing_counts = df.isnull().sum()
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    
    if missing_counts.sum() == 0:
        print("✅ No missing values detected in the dataset")
        return
    
    print("Missing values by column:")
    for col in df.columns:
        if missing_counts[col] > 0:
            print(f"  {col}: {missing_counts[col]} ({missing_percentages[col]:.2f}%)")
    
    total_missing = missing_counts.sum()
    print(f"\nTotal missing values: {total_missing}")
    print(f"Percentage of dataset: {(total_missing / (len(df) * len(df.columns))) * 100:.2f}%")

def analyze_data_quality(df, feature_cols):
    """Analyze overall data quality issues."""
    print("\n" + "="*60)
    print("DATA QUALITY ANALYSIS")
    print("="*60)
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    # Check for constant features
    constant_features = []
    for col in feature_cols:
        if df[col].nunique() <= 1:
            constant_features.append(col)
    
    if constant_features:
        print(f"⚠️  Constant features detected: {constant_features}")
        print("   Recommendation: Remove these features as they provide no information")
    else:
        print("✅ No constant features detected")
    
    # Check for near-constant features (>95% same value)
    near_constant_features = []
    for col in feature_cols:
        mode_frequency = df[col].value_counts().iloc[0] / len(df)
        if mode_frequency > 0.95:
            near_constant_features.append((col, mode_frequency))
    
    if near_constant_features:
        print(f"⚠️  Near-constant features (>95% same value):")
        for feat, freq in near_constant_features:
            print(f"   {feat}: {freq:.3f} frequency")
        print("   Consider removing these features")
    else:
        print("✅ No near-constant features detected")

def generate_recommendations(df, feature_cols, target_col):
    """Generate specific recommendations for data transformation."""
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR MODEL TRAINING")
    print("="*60)
    
    recommendations = []
    
    # Class imbalance recommendations
    class_counts = df[target_col].value_counts()
    imbalance_ratio = class_counts.max() / class_counts.min()
    
    if imbalance_ratio > 3:
        recommendations.append("CRITICAL: Address severe class imbalance using SMOTE, ADASYN, or class weights")
    elif imbalance_ratio > 1.5:
        recommendations.append("MODERATE: Consider oversampling techniques or class weights")
    
    # Feature scaling recommendations
    feature_ranges = {}
    for col in feature_cols:
        feature_ranges[col] = df[col].max() - df[col].min()
    
    max_range = max(feature_ranges.values())
    min_range = min(feature_ranges.values())
    
    if max_range / min_range > 100:
        recommendations.append("SCALING: Features have very different scales - use StandardScaler or RobustScaler")
    
    # Outlier recommendations
    high_outlier_features = []
    for col in feature_cols:
        outliers_iqr, _, _ = detect_outliers_iqr(df[col])
        outlier_percentage = outliers_iqr.sum() / len(df)
        if outlier_percentage > 0.05:  # More than 5% outliers
            high_outlier_features.append((col, outlier_percentage))
    
    if high_outlier_features:
        recommendations.append("OUTLIERS: Consider outlier treatment for features with >5% outliers")
        for feat, pct in high_outlier_features:
            print(f"   {feat}: {pct:.2%} outliers")
    
    # Distribution recommendations
    for col in feature_cols:
        skewness = abs(stats.skew(df[col]))
        if skewness > 2:
            recommendations.append(f"SKEWNESS: Apply log transform or Box-Cox to {col} (skewness: {skewness:.2f})")
    
    # Feature engineering recommendations
    if 'annual_income' in feature_cols and 'num_children' in feature_cols:
        recommendations.append("FEATURE ENGINEERING: Consider creating income_per_person = annual_income / (num_children + 1)")
    
    if 'calories_wk' in feature_cols and 'hrs_exercise_wk' in feature_cols:
        recommendations.append("FEATURE ENGINEERING: Consider creating calories_per_exercise_hour ratio")
    
    if 'exercise_intensity' in feature_cols and 'hrs_exercise_wk' in feature_cols:
        recommendations.append("FEATURE ENGINEERING: Consider creating total_exercise_load = intensity * hours")
    
    # Print all recommendations
    if recommendations:
        print("\nPrioritized recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("✅ Dataset appears well-prepared for model training!")
    
    return recommendations

def main():
    """Main analysis function."""
    print("Enhanced Diabetes Dataset Analysis")
    print("=" * 50)
    
    # Find dataset
    dataset_path = find_dataset_file()
    if not dataset_path:
        print("❌ Error: Could not find diabetes dataset file")
        sys.exit(1)
    
    print(f"\n📁 Loading dataset from: {dataset_path}")
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset loaded successfully")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)
    
    # Identify feature and target columns
    target_col = 'is_diabetic'
    feature_cols = [col for col in df.columns if col != target_col]
    
    print(f"\n🎯 Target column: {target_col}")
    print(f"🔧 Feature columns: {feature_cols}")
    
    # Perform comprehensive analysis
    analyze_missing_values(df)
    analyze_data_quality(df, feature_cols)
    analyze_class_imbalance(df, target_col)
    analyze_feature_distributions(df, feature_cols)
    analyze_feature_correlations(df, feature_cols, target_col)
    
    # Generate recommendations
    recommendations = generate_recommendations(df, feature_cols, target_col)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"📊 Total samples analyzed: {len(df):,}")
    print(f"🔧 Features analyzed: {len(feature_cols)}")
    print(f"💡 Recommendations generated: {len(recommendations)}")
    
    if recommendations:
        print(f"\n⚠️  Action required: {len(recommendations)} improvements identified")
    else:
        print(f"\n✅ Dataset ready for model training!")

if __name__ == "__main__":
    main()
