import pandas as pd
import numpy as np
import os
import json

def min_max_normalize(df, target_column='Discharge'):
    """最小最大归一化 (0-1范围)"""
    params = {}
    normalized_df = df.copy()
    
    # 包含目标列进行归一化
    
    for column in normalized_df.select_dtypes(include=np.number).columns:
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val == min_val:
            normalized_df[column] = 0.5  # 处理常数列
        else:
            normalized_df[column] = (df[column] - min_val) / (max_val - min_val)
        params[column] = {'min': float(min_val), 'max': float(max_val)}
    return normalized_df, params

def z_score_normalize(df):
    """Z得分归一化 (均值0，标准差1)"""
    params = {}
    normalized_df = df.copy()
    target_column = 'Discharge'
    # 包含目标列进行归一化
    features_df = normalized_df.copy()
    
    # Check for non-numeric columns
    non_numeric = features_df.select_dtypes(exclude=np.number).columns
    if len(non_numeric) > 0:
        raise ValueError(f"Non-numeric columns found: {list(non_numeric)}")
    
    for column in features_df.columns:
        # Handle negative values that should be positive (e.g. precipitation)
        if column in ['Prcp', 'Srad', 'Swe']:
            if (features_df[column] < 0).any():
                print(f"Warning: Negative values found in {column} - taking absolute values")
                features_df[column] = features_df[column].abs()
        
        # Calculate statistics
        mean = features_df[column].mean()
        std = features_df[column].std(ddof=0)  # Use population stddev
        print(f"Column {column} - Mean: {mean}, Std: {std}")
        
        # Handle zero variance with epsilon protection
        if np.isclose(std, 0):
            print(f"Warning: Zero variance in {column} - applying epsilon protection")
            epsilon = 1e-8
            normalized_col = (features_df[column] - mean) / (std + epsilon)
        else:
            normalized_col = (features_df[column] - mean) / std
        
        # Check for NaN before assigning
        if normalized_col.isna().any():
            raise ValueError(f"NaN values generated in {column} during normalization")
            
        normalized_df[column] = normalized_col
        params[column] = {'mean': float(mean), 'std': float(std)}
    
    return normalized_df, params

def decimal_scaling_normalize(df):
    """小数定标归一化 (绝对值最大值小于1)"""
    params = {}
    normalized_df = df.copy()
    target_column = 'Discharge'
    
    # 包含目标列进行归一化
    
    for column in normalized_df.select_dtypes(include=np.number).columns:
        max_abs = df[column].abs().max()
        if max_abs == 0:
            j = 0
        else:
            j = np.ceil(np.log10(max_abs))
        scale = 10 ** j
        normalized_df[column] = df[column] / scale
        params[column] = {'scale_factor': float(scale)}  # 已修复类型转换
    return normalized_df, params

def process_dataset(target_column: str = "discharge"):
    """处理dataset目录下所有CSV文件"""
    input_dir = 'dataset'
    methods = [
        (min_max_normalize, 'min_max'),
        (z_score_normalize, 'z_score'),
        (decimal_scaling_normalize, 'decimal_scaling')
    ]
    
    for method, method_name in methods:
        output_dir = os.path.join('preprocessed_data', method_name)
        param_dir = os.path.join('preprocessing_params', method_name)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(param_dir, exist_ok=True)
        
        for filename in os.listdir(input_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(input_dir, filename)
                # Read and parse the fixed-width formatted data
                # Precise column widths based on actual data format:
                # Discharge(9) Dayl(6) Prcp(6) Srad(6) Swe(6) Tmax(6) Tmin(6) Vp(6) Date(10)
                column_widths = [9, 6, 6, 6, 6, 6, 6, 6, 10]  # Sum to total line width
                column_names = ["Discharge", "Dayl", "Prcp", "Srad", "Swe", "Tmax", "Tmin", "Vp", "Date"]
                df = pd.read_fwf(file_path, widths=column_widths, names=column_names)
                
                # Convert all feature columns to numeric
                # Separate target from features
                target_col = "Discharge"
                feature_cols = ["Dayl", "Prcp", "Srad", "Swe", "Tmax", "Tmin", "Vp"]
                for col in feature_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Parse dates with flexible format handling and error tolerance
                df["Date"] = pd.to_datetime(df["Date"], errors='coerce', format='mixed', dayfirst=False)
                
                # Handle invalid dates by filling with forward fill and warning
                invalid_count = df["Date"].isna().sum()
                if invalid_count > 0:
                    print(f"Warning: Found {invalid_count} invalid dates in {filename}, filling with forward fill")
                    df["Date"] = df["Date"].ffill()
                
                # Ensure we have valid dates after cleaning
                df = df.dropna(subset=["Date"])
                
                # Clean remaining NaN and infinite values in features
                df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
                df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0)
                print(f"Processed {filename} - NaN counts after cleaning:")
                print(df[feature_cols].isna().sum())
                print(f"Data types after cleaning:")
                print(df[feature_cols].dtypes)
                print(f"Sample 'Prcp' values:")
                print(df['Prcp'].head())
                df = df.drop(columns=["Date"])  # Remove date column before normalization
                
                # 使用全部数据进行归一化
                normalized_df, params = method(df)
                
                # 保存处理后的数据
                output_path = os.path.join(output_dir, filename)
                normalized_df.to_csv(output_path, index=False)
                
                # 保存归一化参数
                param_path = os.path.join(param_dir, f"{filename}_params.json")
                with open(param_path, 'w') as f:
                    json.dump(params, f)

if __name__ == "__main__":
    process_dataset()
