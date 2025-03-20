import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# 设置中文字体和绘图样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
sns.set(style="whitegrid", palette="muted")

def plot_preprocessing_effects(station_id):
    """可视化单个水文站数据的预处理效果"""
    # 设置文件路径
    raw_path = f'dataset/{station_id}.csv'
    mm_path = f'preprocessed_data/min_max/{station_id}.csv'
    zs_path = f'preprocessed_data/z_score/{station_id}.csv'
    ds_path = f'preprocessed_data/decimal_scaling/{station_id}.csv'
    
    # 检查文件是否存在
    if not all(os.path.exists(p) for p in [raw_path, mm_path, zs_path, ds_path]):
        print(f"跳过 {station_id} - 文件不完整")
        return
    
    try:
        # 加载数据
        raw_df = pd.read_csv(raw_path)
        mm_df = pd.read_csv(mm_path)
        zs_df = pd.read_csv(zs_path)
        ds_df = pd.read_csv(ds_path)
        
        # 选择前4个数值型特征进行可视化
        numeric_cols = raw_df.select_dtypes(include='number').columns.tolist()[:4]
        if len(numeric_cols) < 2:
            print(f"跳过 {station_id} - 有效数据列不足")
            return
            
        # 创建绘图
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'水文站 {station_id} 数据预处理效果对比', fontsize=16)
        
        # 为每个特征创建子图
        for i, col in enumerate(numeric_cols, 1):
            ax_dist = fig.add_subplot(4, 4, i)
            ax_time = fig.add_subplot(4, 4, i+4)
            
            # 绘制分布对比
            sns.kdeplot(raw_df[col], ax=ax_dist, label='原始数据', fill=True)
            sns.kdeplot(mm_df[col], ax=ax_dist, label='最小最大归一化')
            sns.kdeplot(zs_df[col], ax=ax_dist, label='Z得分归一化')
            sns.kdeplot(ds_df[col], ax=ax_dist, label='小数定标归一化')
            ax_dist.set_title(f'{col} 分布对比')
            ax_dist.legend()
            
            # 绘制时序对比（取前100个样本）
            ax_time.plot(raw_df[col].values[:100], label='原始数据')
            ax_time.plot(mm_df[col].values[:100], label='最小最大')
            ax_time.plot(zs_df[col].values[:100], label='Z得分')
            ax_time.plot(ds_df[col].values[:100], label='小数定标')
            ax_time.set_title(f'{col} 时序对比')
            ax_time.legend()
            
        # 调整布局并保存
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_dir = Path('visualization_results')
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / f'{station_id}_preprocessing_effects.png')
        plt.close()
        print(f"已生成 {station_id} 的可视化图表")
        
    except Exception as e:
        print(f"处理 {station_id} 时出错: {str(e)}")

def visualize_all_stations():
    """批量处理所有水文站数据"""
    station_files = [f for f in os.listdir('dataset') if f.endswith('.csv')]
    for i, filename in enumerate(station_files, 1):
        station_id = filename.split('.')[0]
        print(f'正在处理 ({i}/{len(station_files)}) {station_id}')
        plot_preprocessing_effects(station_id)

if __name__ == "__main__":
    visualize_all_stations()
