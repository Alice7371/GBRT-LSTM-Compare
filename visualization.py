import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置中文显示（如果系统支持）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('dataset/01013500.csv', parse_dates=['Date'], dayfirst=False)

# 创建可视化画布
plt.figure(figsize=(16, 20))
sns.set_theme(style="whitegrid", palette="muted")

# 1. 流量和降水时间序列
plt.subplot(4, 2, 1)
sns.lineplot(x='Date', y='Discharge', data=df, color='royalblue')
plt.title('每日流量变化 (2000-2003)')
plt.xlabel('日期')
plt.ylabel('流量 (cms)')

plt.subplot(4, 2, 2)
sns.lineplot(x='Date', y='Prcp', data=df, color='teal')
plt.title('每日降水量变化')
plt.ylabel('降水量 (mm)')

# 2. 温度变化趋势
plt.subplot(4, 2, 3)
sns.lineplot(x='Date', y='Tmax', data=df, color='firebrick')
sns.lineplot(x='Date', y='Tmin', data=df, color='navy')
plt.title('每日最高/最低温度')
plt.ylabel('温度 (°C)')
plt.legend(['最高温','最低温'])

# 3. 变量分布箱线图
plt.subplot(4, 2, 4)
plot_cols = ['Discharge', 'Prcp', 'Tmax', 'Tmin']
sns.boxplot(data=df[plot_cols], orient='h', palette='Set2')
plt.title('主要变量分布')
plt.xlabel('数值范围')

# 4. 变量间相关性热力图
plt.subplot(4, 2, 5)
corr_matrix = df[['Discharge', 'Prcp', 'Srad', 'Tmax', 'Tmin', 'Vp']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('变量间相关系数矩阵')

# 5. 太阳辐射与流量关系
plt.subplot(4, 2, 6)
sns.scatterplot(x='Srad', y='Discharge', data=df, 
                hue=df['Date'].dt.month, palette='viridis', alpha=0.7)
plt.title('太阳辐射与流量关系（按月份着色）')
plt.xlabel('太阳辐射 (W/m²)')

# 调整布局并保存
plt.tight_layout()
plt.savefig('visualization_results.png', dpi=300, bbox_inches='tight')
print("可视化结果已保存至 visualization_results.png")
