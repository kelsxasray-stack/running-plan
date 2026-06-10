import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
sns.set_style("whitegrid")

# 读取数据
df = pd.read_csv('data/running_data.tsv', sep='\t', encoding='utf-8')

# 数据清洗
df['日期'] = pd.to_datetime(df['日期'])
numeric_cols = ['距离', '热量消耗', '平均心率', '最大心率', '有氧效果', '平均步频', 
               '最高步频', '平均步长', '平均垂直摆动', '平均触地时间', '步数']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 处理时间格式
def time_to_seconds(time_str):
    try:
        parts = str(time_str).split(':')
        if len(parts) == 3:
            return int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0])*60 + float(parts[1])
        else:
            return np.nan
    except:
        return np.nan

def pace_to_seconds(pace_str):
    try:
        parts = str(pace_str).split(':')
        if len(parts) == 2:
            return int(parts[0])*60 + int(parts[1])
        else:
            return np.nan
    except:
        return np.nan

df['时间_秒'] = df['时间'].apply(time_to_seconds)
df['平均配速_秒'] = df['平均速度'].apply(pace_to_seconds)
df = df.sort_values('日期').reset_index(drop=True)

# 创建输出目录
import os
os.makedirs('output', exist_ok=True)

print("🎨 正在生成9张专业可视化图表...\n")

# ============================================
# 1. 里程趋势图
# ============================================
print("✅ 生成图表 1/9: 📈 里程趋势图")
fig, ax = plt.subplots(figsize=(14, 6))
df_sorted = df.set_index('日期').sort_index()
ax.plot(df_sorted.index, df_sorted['距离'], color='#FF6B6B', linewidth=2.5, marker='o', markersize=4, alpha=0.8)
ax.fill_between(df_sorted.index, df_sorted['距离'], alpha=0.3, color='#FF6B6B')
ax.set_title('📈 每次训练距离趋势分析', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('距离 (km)', fontsize=12, fontweight='bold')
ax.set_xlabel('日期', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.axhline(y=df['距离'].mean(), color='red', linestyle='--', linewidth=2, label=f'平均: {df["距离"].mean():.2f}km', alpha=0.7)
ax.legend(fontsize=10)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('output/01_里程趋势图.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 2. 月度统计柱状图
# ============================================
print("✅ 生成图表 2/9: 📊 月度统计柱状图")
df['年月'] = df['日期'].dt.to_period('M')
monthly = df.groupby('年月').agg({
    '距离': 'sum',
    '热量消耗': 'sum',
    '日期': 'count'
}).rename(columns={'日期': '次数'})

fig, ax1 = plt.subplots(figsize=(14, 7))
x = np.arange(len(monthly))
width = 0.35

bars1 = ax1.bar(x - width/2, monthly['距离'], width, label='总里程 (km)', color='#4ECDC4', edgecolor='black', linewidth=1.5)
ax1.set_ylabel('里程 (km)', fontsize=12, fontweight='bold', color='#4ECDC4')
ax1.tick_params(axis='y', labelcolor='#4ECDC4')

ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, monthly['次数'], width, label='训练次数', color='#95E1D3', edgecolor='black', linewidth=1.5)
ax2.set_ylabel('训练次数', fontsize=12, fontweight='bold', color='#95E1D3')
ax2.tick_params(axis='y', labelcolor='#95E1D3')

ax1.set_xlabel('月份', fontsize=12, fontweight='bold')
ax1.set_title('📊 月度训练统计 - 里程与频次', fontsize=16, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels([str(month) for month in monthly.index], rotation=45, ha='right')
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

# 添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('output/02_月度统计柱状图.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 3. 活动类型分布饼图
# ============================================
print("✅ 生成图表 3/9: 🥧 活动类型分布饼图")
activity_dist = df.groupby('活动类型')['距离'].sum()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F8B500', '#95E1D3']
colors = colors[:len(activity_dist)]

# 按里程分布的饼图
wedges1, texts1, autotexts1 = ax1.pie(activity_dist, labels=activity_dist.index, autopct='%1.1f%%',
                                       colors=colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax1.set_title('🥧 活动类型分布 (按里程)', fontsize=14, fontweight='bold', pad=20)

# 按训练次数分布的饼图
activity_count = df.groupby('活动类型')['日期'].count()
wedges2, texts2, autotexts2 = ax2.pie(activity_count, labels=activity_count.index, autopct='%1.1f%%',
                                       colors=colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax2.set_title('🥧 活动类型分布 (按训练次数)', fontsize=14, fontweight='bold', pad=20)

# 美化饼图
for autotext in autotexts1 + autotexts2:
    autotext.set_color('white')
    autotext.set_fontsize(10)
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.savefig('output/03_活动类型分布饼图.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 4. 心率分布直方图
# ============================================
print("✅ 生成图表 4/9: 💓 心率分布直方图")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 平均心率分布
ax1.hist(df['平均心率'].dropna(), bins=40, color='#FF6B6B', alpha=0.7, edgecolor='black', linewidth=1.2)
ax1.axvline(df['平均心率'].mean(), color='red', linestyle='--', linewidth=2.5, label=f'平均: {df["平均心率"].mean():.0f} bpm')
ax1.axvline(df['平均心率'].median(), color='orange', linestyle='--', linewidth=2.5, label=f'中位数: {df["平均心率"].median():.0f} bpm')
ax1.set_title('💓 平均心率分布', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('心率 (bpm)', fontsize=12, fontweight='bold')
ax1.set_ylabel('频数', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

# 最大心率分布
ax2.hist(df['最大心率'].dropna(), bins=40, color='#45B7D1', alpha=0.7, edgecolor='black', linewidth=1.2)
ax2.axvline(df['最大心率'].mean(), color='blue', linestyle='--', linewidth=2.5, label=f'平均: {df["最大心率"].mean():.0f} bpm')
ax2.axvline(df['最大心率'].median(), color='green', linestyle='--', linewidth=2.5, label=f'中位数: {df["最大心率"].median():.0f} bpm')
ax2.set_title('💓 最大心率分布', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('心率 (bpm)', fontsize=12, fontweight='bold')
ax2.set_ylabel('频数', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('output/04_心率分布直方图.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 5. 配速与距离关系散点图
# ============================================
print("✅ 生成图表 5/9: 🎯 配速与距离关系图")
fig, ax = plt.subplots(figsize=(14, 8))
scatter = ax.scatter(df['距离'], df['平均配速_秒']/60, 
                     c=df['平均心率'], cmap='RdYlGn_r', s=150, alpha=0.6, 
                     edgecolor='black', linewidth=1.5)
ax.set_title('🎯 配速与距离的关系分析 (颜色表示心率)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('距离 (km)', fontsize=12, fontweight='bold')
ax.set_ylabel('平均配速 (min/km)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')

# 添加趋势线
z = np.polyfit(df['距离'].dropna(), (df['平均配速_秒']/60).dropna(), 2)
p = np.poly1d(z)
x_trend = np.linspace(df['距离'].min(), df['距离'].max(), 100)
ax.plot(x_trend, p(x_trend), "r--", linewidth=2.5, alpha=0.8, label='趋势线')

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('平均心率 (bpm)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('output/05_配速与距离关系图.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 6. 月度训练次数统计
# ============================================
print("✅ 生成图表 6/9: 📅 训练频率统计")
monthly_counts = df.groupby(df['日期'].dt.to_period('M')).size()
fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(range(len(monthly_counts)), monthly_counts.values, color='#95E1D3', 
              edgecolor='black', linewidth=1.5, alpha=0.8)
ax.set_title('📅 月度训练次数统计', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('训练次数', fontsize=12, fontweight='bold')
ax.set_xlabel('月份', fontsize=12, fontweight='bold')
ax.set_xticks(range(len(monthly_counts)))
ax.set_xticklabels([str(month) for month in monthly_counts.index], rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

# 添加数值标签
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.axhline(y=monthly_counts.mean(), color='red', linestyle='--', linewidth=2, 
          label=f'平均: {monthly_counts.mean():.1f}次/月', alpha=0.7)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('output/06_训练频率统计.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 7. 热量消耗趋势 (7日移动平均)
# ============================================
print("✅ 生成图表 7/9: 🔥 热量消耗趋势")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# 原始热量消耗
df_sorted = df.set_index('日期').sort_index()
ax1.scatter(df_sorted.index, df_sorted['热量消耗'], color='#FF6B6B', s=80, alpha=0.6, edgecolor='black', label='单次训练')
ax1.plot(df_sorted.index, df_sorted['热量消耗'].rolling(window=7).mean(), 
        color='#F8B500', linewidth=3, label='7日移动平均', alpha=0.9)
ax1.fill_between(df_sorted.index, df_sorted['热量消耗'], alpha=0.2, color='#FF6B6B')
ax1.set_title('🔥 热量消耗趋势分析 (单次与7日平均)', fontsize=16, fontweight='bold', pad=20)
ax1.set_ylabel('热量 (kcal)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=10)

# 月度热量总计
monthly_calories = df.groupby('年月')['热量消耗'].sum()
ax2.bar(range(len(monthly_calories)), monthly_calories.values, 
       color='#FF6B6B', edgecolor='black', linewidth=1.5, alpha=0.7)
ax2.plot(range(len(monthly_calories)), monthly_calories.values, 
        color='#F8B500', linewidth=2.5, marker='o', markersize=8, alpha=0.8)
ax2.set_title('🔥 月度累计热量消耗', fontsize=16, fontweight='bold', pad=20)
ax2.set_ylabel('热量 (kcal)', fontsize=12, fontweight='bold')
ax2.set_xlabel('月份', fontsize=12, fontweight='bold')
ax2.set_xticks(range(len(monthly_calories)))
ax2.set_xticklabels([str(month) for month in monthly_calories.index], rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

# 添加数值标签
for i, val in enumerate(monthly_calories.values):
    ax2.text(i, val, f'{int(val)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('output/07_热量消耗趋势.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 8. 训练时段分布分析
# ============================================
print("✅ 生成图表 8/9: ⏰ 训练时段分析")
df['小时'] = df['日期'].dt.hour
df['星期'] = df['日期'].dt.day_name()
hour_counts = df['小时'].value_counts().sort_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 按小时统计
bars = ax1.bar(hour_counts.index, hour_counts.values, color='#A8E6CF', edgecolor='black', linewidth=1.5, alpha=0.8)
ax1.set_title('⏰ 训练时段分布 (按小时)', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('小时 (24小时制)', fontsize=12, fontweight='bold')
ax1.set_ylabel('训练次数', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    if height > 0:
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 按星期统计
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_counts = df['星期'].value_counts().reindex(day_order)
day_names_cn = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
colors_days = ['#FF6B6B' if day in ['Saturday', 'Sunday'] else '#4ECDC4' for day in day_order]

bars2 = ax2.bar(day_names_cn, day_counts.values, color=colors_days, edgecolor='black', linewidth=1.5, alpha=0.8)
ax2.set_title('⏰ 训练日期分布 (按星期)', fontsize=14, fontweight='bold', pad=15)
ax2.set_ylabel('训练次数', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

# 添加数值标签
for bar in bars2:
    height = bar.get_height()
    if height > 0:
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('output/08_训练时段分析.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 9. 周度统计
# ============================================
print("✅ 生成图表 9/9: 📊 周度统计")
df['周'] = df['日期'].dt.isocalendar().week
df['年'] = df['日期'].dt.year
weekly_stats = df.groupby(['年', '周']).agg({
    '距离': 'sum',
    '热量消耗': 'sum',
    '日期': 'count'
}).rename(columns={'日期': '次数'})

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))

# 周度里程
ax1.plot(range(len(weekly_stats)), weekly_stats['距离'].values, 
        color='#FF6B6B', linewidth=2.5, marker='o', markersize=6, alpha=0.8)
ax1.fill_between(range(len(weekly_stats)), weekly_stats['距离'].values, alpha=0.3, color='#FF6B6B')
ax1.set_title('📊 周度总里程统计', fontsize=14, fontweight='bold', pad=15)
ax1.set_ylabel('里程 (km)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.axhline(y=weekly_stats['距离'].mean(), color='red', linestyle='--', linewidth=2, 
           label=f'平均: {weekly_stats["距离"].mean():.1f}km/周', alpha=0.7)
ax1.legend(fontsize=10)

# 周度热量
ax2.bar(range(len(weekly_stats)), weekly_stats['热量消耗'].values, 
       color='#F8B500', edgecolor='black', linewidth=1.5, alpha=0.7)
ax2.set_title('📊 周度热量消耗统计', fontsize=14, fontweight='bold', pad=15)
ax2.set_ylabel('热量 (kcal)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

# 周度训练次数
ax3.bar(range(len(weekly_stats)), weekly_stats['次数'].values, 
       color='#95E1D3', edgecolor='black', linewidth=1.5, alpha=0.7)
ax3.set_title('📊 周度训练次数统计', fontsize=14, fontweight='bold', pad=15)
ax3.set_xlabel('周数', fontsize=12, fontweight='bold')
ax3.set_ylabel('次数', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
ax3.axhline(y=weekly_stats['次数'].mean(), color='red', linestyle='--', linewidth=2, 
           label=f'平均: {weekly_stats["次数"].mean():.1f}次/周', alpha=0.7)
ax3.legend(fontsize=10)

plt.tight_layout()
plt.savefig('output/09_周度统计.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 汇总报告
# ============================================
print("\n" + "="*60)
print("✅ 所有9张图表已成功生成！")
print("="*60)
print("\n📊 生成的图表清单:")
print("  1️⃣  01_里程趋势图.png - 训练距离随时间变化趋势")
print("  2️⃣  02_月度统计柱状图.png - 月度里程和训练频次")
print("  3️⃣  03_活动类型分布饼图.png - 不同训练方式比例")
print("  4️⃣  04_心率分布直方图.png - 心率数据统计分析")
print("  5️⃣  05_配速与距离关系图.png - 配速和距离的相关性")
print("  6️⃣  06_训练频率统计.png - 每月训练次数统计")
print("  7️⃣  07_热量消耗趋势.png - 热量消耗变化趋势")
print("  8️⃣  08_训练时段分析.png - 按时间和星期分布")
print("  9️⃣  09_周度统计.png - 周度里程、热量和频次")
print("\n📁 所有图表保存在: output/ 目录")
print("🎨 分辨率: 300 DPI (高清质量)")
print("💾 格式: PNG (支持透明背景)")

# 数据统计信息
print("\n" + "="*60)
print("📈 数据统计摘要")
print("="*60)
print(f"总训练次数: {len(df)} 次")
print(f"总里程: {df['距离'].sum():.2f} km")
print(f"总热量消耗: {df['热量消耗'].sum():.0f} kcal")
print(f"平均配速: {df['平均配速_秒'].mean()/60:.2f} min/km")
print(f"最快配速: {df['平均配速_秒'].min()/60:.2f} min/km")
print(f"平均心率: {df['平均心率'].mean():.0f} bpm")
print(f"最大心率: {df['最大心率'].max():.0f} bpm")
print(f"日期范围: {df['日期'].min().date()} 至 {df['日期'].max().date()}")
print("="*60)
