import pandas as pd
import numpy as np
import os
import json

def min_max_normalize(df):
    """最小最大归一化 (0-1范围)"""
    params = {}
    normalized_df = df.copy()
    for column in df.select_dtypes(include=np.number).columns:
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
    for column in df.select_dtypes(include=np.number).columns:
        mean = df[column].mean()
        std = df[column].std()
        if std == 0:
            normalized_df[column] = 0  # 处理常数列
        else:
            normalized_df[column] = (df[column] - mean) / std
        params[column] = {'mean': float(mean), 'std': float(std)}
    return normalized_df, params

def decimal_scaling_normalize(df):
    """小数定标归一化 (绝对值最大值小于1)"""
    params = {}
    normalized_df = df.copy()
    for column in df.select_dtypes(include=np.number).columns:
        max_abs = df[column].abs().max()
        if max_abs == 0:
            j = 0
        else:
            j = np.ceil(np.log10(max_abs))
        scale = 10 ** j
        normalized_df[column] = df[column] / scale
        params[column] = {'scale_factor': float(scale)}  # 已修复类型转换
    return normalized_df, params

def process_dataset():
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
                df = pd.read_csv(file_path)
                
                # 应用归一化方法
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
