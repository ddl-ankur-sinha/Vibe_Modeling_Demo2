#!/usr/bin/env python3
"""
Enhanced Diabetes Dataset Transformation Script
This script applies improved data engineering techniques based on comprehensive analysis
to better prepare the dataset for model training.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
import joblib
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

def create_engineered_features(df):
    """Create new engineered features that may improve model performance."""
    print("\n🔧 Creating engineered features...")
    
    df_engineered = df.copy()
    
    # 1. Income per person (accounting for family size)
    df_engineered['income_per_person'] = df_engineered['annual_income'] / (df_engineered['num_children'] + 1)
    
    # 2. Calories per exercise hour (metabolic efficiency)
    # Add small constant to avoid division by zero
    df_engineered['calories_per_exercise_hour'] = df_engineered['calories_wk'] / (df_engineered['hrs_exercise_wk'] + 0.01)
    
    # 3. Total exercise load (intensity * hours)
    df_engineered['total_exercise_load'] = df_engineered['exercise_intensity'] * df_engineered['hrs_exercise_wk']
    
    # 4. BMI category approximation (using weight as proxy for BMI)
    # Create weight categories based on distribution
    weight_q25, weight_q75 = df_engineered['weight'].quantile([0.25, 0.75])
    df_engineered['weight_category'] = pd.cut(
        df_engineered['weight'], 
        bins=[-np.inf, weight_q25, weight_q75, np.inf], 
        labels=[0, 1, 2]  # 0: low, 1: normal, 2: high
    ).astype(int)
    
    # 5. Exercise frequency indicator (high/low exercise)
    exercise_median = df_engineered['hrs_exercise_wk'].median()
    df_engineered['high_exercise'] = (df_engineered['hrs_exercise_wk'] > exercise_median).astype(int)
    
    # 6. Calorie intake category
    calorie_q33, calorie_q67 = df_engineered['calories_wk'].quantile([0.33, 0.67])
    df_engineered['calorie_category'] = pd.cut(
        df_engineered['calories_wk'],
        bins=[-np.inf, calorie_q33, calorie_q67, np.inf],
        labels=[0, 1, 2]  # 0: low, 1: moderate, 2: high
    ).astype(int)
    
    # 7. Financial stress indicator (low income with children)
    income_q25 = df_engineered['annual_income'].quantile(0.25)
    df_engineered['financial_stress'] = (
        (df_engineered['annual_income'] <= income_q25) & 
        (df_engineered['num_children'] > 0)
    ).astype(int)
    
    new_features = [
        'income_per_person', 'calories_per_exercise_hour', 'total_exercise_load',
        'weight_category', 'high_exercise', 'calorie_category', 'financial_stress'
    ]
    
    print(f"   Created {len(new_features)} new features:")
    for feat in new_features:
        print(f"   - {feat}")
    
    return df_engineered, new_features

def handle_outliers(df, feature_cols, method='iqr_clip', iqr_multiplier=1.5):
    """Handle outliers using specified method."""
    print(f"\n🔧 Handling outliers using {method} method...")
    
    df_cleaned = df.copy()
    outlier_stats = {}
    
    for col in feature_cols:
        original_count = len(df_cleaned)
        
        if method == 'iqr_clip':
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            
            # Count outliers before clipping
            outliers_before = ((df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)).sum()
            
            # Clip outliers
            df_cleaned[col] = np.clip(df_cleaned[col], lower_bound, upper_bound)
            
            outlier_stats[col] = {
                'method': 'IQR clipping',
                'outliers_found': outliers_before,
                'percentage': (outliers_before / original_count) * 100
            }
            
        elif method == 'z_score_remove':
            z_scores = np.abs((df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std())
            outliers_mask = z_scores <= 3
            outliers_before = (~outliers_mask).sum()
            
            df_cleaned = df_cleaned[outliers_mask]
            
            outlier_stats[col] = {
                'method': 'Z-score removal',
                'outliers_found': outliers_before,
                'percentage': (outliers_before / original_count) * 100
            }
    
    # Print outlier statistics
    for col, stats in outlier_stats.items():
        if stats['outliers_found'] > 0:
            print(f"   {col}: {stats['outliers_found']} outliers ({stats['percentage']:.2f}%) handled via {stats['method']}")
        else:
            print(f"   {col}: No outliers detected")
    
    return df_cleaned, outlier_stats

def apply_power_transforms(df, feature_cols, target_col):
    """Apply power transformations to reduce skewness."""
    print(f"\n🔧 Applying power transformations to reduce skewness...")
    
    df_transformed = df.copy()
    transformers = {}
    
    from scipy import stats
    
    for col in feature_cols:
        # Calculate skewness
        skewness = stats.skew(df[col])
        
        if abs(skewness) > 1.0:  # Significantly skewed
            print(f"   {col}: skewness = {skewness:.3f} - applying transformation")
            
            # Handle negative or zero values by adding a constant
            min_val = df[col].min()
            if min_val <= 0:
                df_transformed[col] = df[col] - min_val + 1
            
            try:
                # Apply Yeo-Johnson transformation (handles negative values)
                transformer = PowerTransformer(method='yeo-johnson', standardize=False)
                df_transformed[col] = transformer.fit_transform(df_transformed[[col]]).ravel()
                transformers[col] = transformer
                
                # Check new skewness
                new_skewness = stats.skew(df_transformed[col])
                print(f"     → New skewness: {new_skewness:.3f}")
                
            except Exception as e:
                print(f"     → Transformation failed for {col}: {e}")
                df_transformed[col] = df[col]  # Keep original
        else:
            print(f"   {col}: skewness = {skewness:.3f} - no transformation needed")
    
    return df_transformed, transformers

def balance_classes(X, y, method='smote', random_state=42):
    """Apply class balancing techniques to address imbalance."""
    print(f"\n⚖️  Balancing classes using {method} method...")
    
    # Check original distribution
    original_dist = pd.Series(y).value_counts().sort_index()
    print(f"   Original distribution:")
    for class_val, count in original_dist.items():
        print(f"     Class {class_val}: {count} samples")
    
    imbalance_ratio = original_dist.max() / original_dist.min()
    print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio < 1.5:
        print("   Classes are relatively balanced - skipping resampling")
        return X, y
    
    try:
        if method == 'smote':
            resampler = SMOTE(random_state=random_state, k_neighbors=3)
        elif method == 'adasyn':
            resampler = ADASYN(random_state=random_state, n_neighbors=3)
        elif method == 'smote_tomek':
            resampler = SMOTETomek(random_state=random_state)
        elif method == 'smote_enn':
            resampler = SMOTEENN(random_state=random_state)
        elif method == 'undersample':
            resampler = RandomUnderSampler(random_state=random_state)
        else:
            print(f"   Unknown method {method} - skipping resampling")
            return X, y
        
        X_resampled, y_resampled = resampler.fit_resample(X, y)
        
        # Check new distribution
        new_dist = pd.Series(y_resampled).value_counts().sort_index()
        print(f"   New distribution:")
        for class_val, count in new_dist.items():
            print(f"     Class {class_val}: {count} samples")
        
        new_imbalance_ratio = new_dist.max() / new_dist.min()
        print(f"   New imbalance ratio: {new_imbalance_ratio:.2f}:1")
        
        return X_resampled, y_resampled
        
    except Exception as e:
        print(f"   Resampling failed: {e}")
        print("   Returning original data")
        return X, y

def scale_features(X_train, X_val, method='robust'):
    """Apply feature scaling to training and validation sets."""
    print(f"\n📏 Scaling features using {method} scaler...")
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        print(f"   Unknown scaling method {method} - using robust scaler")
        scaler = RobustScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert back to DataFrames with column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    
    print(f"   Scaled {X_train.shape[1]} features")
    
    return X_train_scaled, X_val_scaled, scaler

def main():
    """Main transformation function."""
    print("Enhanced Diabetes Dataset Transformation")
    print("=" * 50)
    
    # Configuration
    config = {
        'random_state': 42,
        'test_size': 0.2,
        'outlier_method': 'iqr_clip',  # 'iqr_clip' or 'z_score_remove'
        'scaling_method': 'robust',    # 'standard' or 'robust'
        'class_balance_method': 'smote', # 'smote', 'adasyn', 'smote_tomek', 'undersample', 'none'
        'apply_power_transforms': True,
        'create_features': True
    }
    
    print("\n📋 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Find and load dataset
    dataset_path = find_dataset_file()
    if not dataset_path:
        print("❌ Error: Could not find diabetes dataset file")
        sys.exit(1)
    
    print(f"\n📁 Loading dataset from: {dataset_path}")
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset loaded successfully")
        print(f"   Original shape: {df.shape}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)
    
    # Identify columns
    target_col = 'is_diabetic'
    original_features = [col for col in df.columns if col != target_col]
    
    print(f"\n🎯 Target column: {target_col}")
    print(f"🔧 Original features: {original_features}")
    
    # Step 1: Create engineered features
    if config['create_features']:
        df, new_features = create_engineered_features(df)
        all_features = original_features + new_features
    else:
        all_features = original_features
        new_features = []
    
    print(f"\n📊 Total features after engineering: {len(all_features)}")
    
    # Step 2: Handle outliers
    df, outlier_stats = handle_outliers(df, all_features, method=config['outlier_method'])
    
    # Step 3: Apply power transformations
    if config['apply_power_transforms']:
        df, transformers = apply_power_transforms(df, all_features, target_col)
    else:
        transformers = {}
    
    print(f"\n📊 Dataset shape after preprocessing: {df.shape}")
    
    # Step 4: Split features and target
    X = df[all_features]
    y = df[target_col]
    
    # Step 5: Train-validation split
    print(f"\n✂️  Splitting dataset (test_size={config['test_size']})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=config['test_size'], 
        random_state=config['random_state'],
        stratify=y
    )
    
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Validation set: {X_val.shape[0]} samples")
    
    # Step 6: Apply class balancing (only to training set)
    if config['class_balance_method'] != 'none':
        X_train, y_train = balance_classes(
            X_train, y_train, 
            method=config['class_balance_method'],
            random_state=config['random_state']
        )
    
    # Step 7: Feature scaling
    X_train_scaled, X_val_scaled, scaler = scale_features(
        X_train, X_val, 
        method=config['scaling_method']
    )
    
    # Step 8: Save processed datasets
    print(f"\n💾 Saving processed datasets...")
    
    # Create full datasets with target
    train_df = X_train_scaled.copy()
    train_df[target_col] = y_train
    
    val_df = X_val_scaled.copy()
    val_df[target_col] = y_val
    
    # Save files
    output_files = []
    
    # Enhanced training set
    train_file = 'diabetes_dataset_train_enhanced.csv'
    train_df.to_csv(train_file, index=False)
    output_files.append(train_file)
    print(f"   ✅ {train_file}")
    
    # Enhanced validation set
    val_file = 'diabetes_dataset_val_enhanced.csv'
    val_df.to_csv(val_file, index=False)
    output_files.append(val_file)
    print(f"   ✅ {val_file}")
    
    # Save preprocessing objects
    scaler_file = 'diabetes_scaler_enhanced.joblib'
    joblib.dump(scaler, scaler_file)
    output_files.append(scaler_file)
    print(f"   ✅ {scaler_file}")
    
    if transformers:
        transformers_file = 'diabetes_transformers_enhanced.joblib'
        joblib.dump(transformers, transformers_file)
        output_files.append(transformers_file)
        print(f"   ✅ {transformers_file}")
    
    # Step 9: Generate transformation summary
    print(f"\n" + "="*60)
    print("TRANSFORMATION SUMMARY")
    print("="*60)
    
    print(f"📊 Dataset Statistics:")
    print(f"   Original samples: {len(df):,}")
    print(f"   Final training samples: {len(train_df):,}")
    print(f"   Final validation samples: {len(val_df):,}")
    print(f"   Original features: {len(original_features)}")
    print(f"   Engineered features: {len(new_features)}")
    print(f"   Total features: {len(all_features)}")
    
    print(f"\n🔧 Transformations Applied:")
    if config['create_features']:
        print(f"   ✅ Feature engineering: {len(new_features)} new features")
    if outlier_stats:
        total_outliers = sum(stats['outliers_found'] for stats in outlier_stats.values())
        print(f"   ✅ Outlier handling: {total_outliers} outliers processed")
    if transformers:
        print(f"   ✅ Power transformations: {len(transformers)} features transformed")
    if config['class_balance_method'] != 'none':
        print(f"   ✅ Class balancing: {config['class_balance_method']} applied")
    print(f"   ✅ Feature scaling: {config['scaling_method']} scaler applied")
    
    print(f"\n📁 Output Files:")
    for file in output_files:
        print(f"   - {file}")
    
    # Final class distribution
    print(f"\n⚖️  Final Class Distribution:")
    train_dist = pd.Series(y_train).value_counts().sort_index()
    val_dist = pd.Series(y_val).value_counts().sort_index()
    
    print(f"   Training set:")
    for class_val, count in train_dist.items():
        pct = count / len(y_train) * 100
        print(f"     Class {class_val}: {count} samples ({pct:.1f}%)")
    
    print(f"   Validation set:")
    for class_val, count in val_dist.items():
        pct = count / len(y_val) * 100
        print(f"     Class {class_val}: {count} samples ({pct:.1f}%)")
    
    print(f"\n✅ Enhanced dataset transformation completed successfully!")
    print(f"📈 Ready for improved model training with {len(all_features)} features")

if __name__ == "__main__":
    main()
