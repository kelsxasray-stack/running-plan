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
sns.set_style("whitegrid")

class RunningAnalyzer:
    """跑步数据分析类"""
    
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, sep='\t', encoding='utf-8')
        self.clean_data()
    
    def clean_data(self):
        """数据清洗"""
        # 转换日期格式
        self.df['日期'] = pd.to_datetime(self.df['日期'])
        
        # 处理数字列
        numeric_cols = ['距离', '热量消耗', '平均心率', '最大心率', '有氧效果', '平均步频', 
                       '最高步频', '平均步长', '平均垂直摆动', '平均触地时间', '步数']
        
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # 处理时间格式 (mm:ss 或 h:mm:ss -> 秒)
        self.df['时间_秒'] = self.df['时间'].apply(self._time_to_seconds)
        
        # 提取配速数据
        self.df['平均配速_秒'] = self.df['平均速度'].apply(self._pace_to_seconds)
        
        # 按日期排序
        self.df = self.df.sort_values('日期').reset_index(drop=True)
    
    @staticmethod
    def _time_to_seconds(time_str):
        """将时间字符串转换为秒"""
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
    
    @staticmethod
    def _pace_to_seconds(pace_str):
        """将配速字符串转换为秒/km"""
        try:
            parts = str(pace_str).split(':')
            if len(parts) == 2:
                return int(parts[0])*60 + int(parts[1])
            else:
                return np.nan
        except:
            return np.nan
    
    def get_statistics(self):
        """获取统计数据"""
        stats = {
            '总记录数': len(self.df),
            '总里程(km)': self.df['距离'].sum(),
            '总时间(小时)': self.df['时间_秒'].sum() / 3600,
            '平均单次距离(km)': self.df['距离'].mean(),
            '最长单次距离(km)': self.df['距离'].max(),
            '平均心率(bpm)': self.df['平均心率'].mean(),
            '最大心率(bpm)': self.df['最大心率'].max(),
            '总热量消耗(kcal)': self.df['热量消耗'].sum(),
            '平均配速(min/km)': self.df['平均配速_秒'].mean() / 60,
            '最佳配速(min/km)': self.df['平均配速_秒'].min() / 60,
        }
        return stats
    
    def get_monthly_stats(self):
        """获取月度统计"""
        self.df['年月'] = self.df['日期'].dt.to_period('M')
        monthly = self.df.groupby('年月').agg({
            '距离': 'sum',
            '时间_秒': 'sum',
            '热量消耗': 'sum',
            '日期': 'count'
        }).rename(columns={'日期': '次数'})
        
        monthly['时间_小时'] = monthly['时间_秒'] / 3600
        monthly['平均配速'] = (monthly['时间_小时'] * 60) / monthly['距离']
        
        return monthly
    
    def get_activity_type_stats(self):
        """获取活动类型统计"""
        return self.df.groupby('活动类型').agg({
            '距离': 'sum',
            '热量消耗': 'sum',
            '日期': 'count'
        }).rename(columns={'日期': '次数'})
    
    def generate_report(self):
        """生成分析报告"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("🏃 跑步记录数据分析报告")
        print("="*60 + "\n")
        
        print("📊 总体统计")
        print("-" * 60)
        for key, value in stats.items():
            if '小时' in key or '秒' in key or 'min' in key:
                print(f"{key:20s}: {value:10.2f}")
            else:
                print(f"{key:20s}: {value:10.2f}")
        
        print("\n📅 月度统计")
        print("-" * 60)
        monthly = self.get_monthly_stats()
        print(monthly)
        
        print("\n🏷️ 活动类型统计")
        print("-" * 60)
        activity = self.get_activity_type_stats()
        print(activity)
        
        print("\n🎯 个人纪录")
        print("-" * 60)
        max_distance = self.df.loc[self.df['距离'].idxmax()]
        print(f"最长距离: {max_distance['距离']:.2f}km - {max_distance['日期'].date()} ({max_distance['标题']})")
        
        max_calories = self.df.loc[self.df['热量消耗'].idxmax()]
        print(f"最多热量: {max_calories['热量消耗']:.0f}kcal - {max_calories['日期'].date()}")
        
        best_pace = self.df.loc[self.df['平均配速_秒'].idxmin()]
        print(f"最快配速: {best_pace['平均速度']} (min/km) - {best_pace['日期'].date()}")
        
        return stats, monthly, activity
    
    def create_visualizations(self, output_dir='./output'):
        """创建可视化图表"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建大图表
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 里程趋势图
        ax1 = plt.subplot(3, 3, 1)
        self.df.set_index('日期')['距离'].plot(ax=ax1, color='#FF6B6B', linewidth=2, marker='o', markersize=3)
        ax1.set_title('📈 每次训练距离趋势', fontsize=12, fontweight='bold')
        ax1.set_ylabel('距离 (km)')
        ax1.grid(True, alpha=0.3)
        
        # 2. 月度里程统计
        ax2 = plt.subplot(3, 3, 2)
        monthly = self.get_monthly_stats()
        monthly['距离'].plot(kind='bar', ax=ax2, color='#4ECDC4', edgecolor='black')
        ax2.set_title('📊 月度总里程统计', fontsize=12, fontweight='bold')
        ax2.set_ylabel('里程 (km)')
        ax2.set_xlabel('月份')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. 活动类型分布
        ax3 = plt.subplot(3, 3, 3)
        activity = self.get_activity_type_stats()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        activity['距离'].plot(kind='pie', ax=ax3, autopct='%1.1f%%', colors=colors)
        ax3.set_title('🏷️ 活动类型分布 (按里程)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('')
        
        # 4. 心率分布直方���
        ax4 = plt.subplot(3, 3, 4)
        ax4.hist(self.df['平均心率'].dropna(), bins=30, color='#FF6B6B', alpha=0.7, edgecolor='black')
        ax4.set_title('💓 平均心率分布', fontsize=12, fontweight='bold')
        ax4.set_xlabel('心率 (bpm)')
        ax4.set_ylabel('频数')
        ax4.axvline(self.df['平均心率'].mean(), color='red', linestyle='--', linewidth=2, label=f'平均: {self.df["平均心率"].mean():.0f}')
        ax4.legend()
        
        # 5. 配速与距离散点图
        ax5 = plt.subplot(3, 3, 5)
        scatter = ax5.scatter(self.df['距离'], self.df['平均配速_秒']/60, 
                             c=self.df['平均心率'], cmap='RdYlGn_r', s=100, alpha=0.6, edgecolor='black')
        ax5.set_title('🎯 配速与距离关系', fontsize=12, fontweight='bold')
        ax5.set_xlabel('距离 (km)')
        ax5.set_ylabel('平均配速 (min/km)')
        plt.colorbar(scatter, ax=ax5, label='平均心率 (bpm)')
        
        # 6. 月度训练次数
        ax6 = plt.subplot(3, 3, 6)
        monthly_counts = self.df.groupby(self.df['日期'].dt.to_period('M')).size()
        monthly_counts.plot(kind='bar', ax=ax6, color='#95E1D3', edgecolor='black')
        ax6.set_title('📅 月度训练次数', fontsize=12, fontweight='bold')
        ax6.set_ylabel('次数')
        ax6.set_xlabel('月份')
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
        
        # 7. 热量消耗趋势
        ax7 = plt.subplot(3, 3, 7)
        self.df.set_index('日期')['热量消耗'].rolling(window=7).mean().plot(ax=ax7, color='#F8B500', linewidth=2)
        ax7.set_title('🔥 热量消耗趋势 (7日移动平均)', fontsize=12, fontweight='bold')
        ax7.set_ylabel('热量 (kcal)')
        ax7.grid(True, alpha=0.3)
        
        # 8. 时间分布
        ax8 = plt.subplot(3, 3, 8)
        self.df['小时'] = self.df['日期'].dt.hour
        hour_counts = self.df['小时'].value_counts().sort_index()
        ax8.bar(hour_counts.index, hour_counts.values, color='#A8E6CF', edgecolor='black')
        ax8.set_title('⏰ 训练时段分布', fontsize=12, fontweight='bold')
        ax8.set_xlabel('小时')
        ax8.set_ylabel('训练次数')
        
        # 9. 周度统计
        ax9 = plt.subplot(3, 3, 9)
        self.df['周'] = self.df['日期'].dt.isocalendar().week
        self.df['年'] = self.df['日期'].dt.year
        weekly = self.df.groupby(['年', '周'])['距离'].sum()
        weekly.plot(ax=ax9, color='#FFB6C1', linewidth=2, marker='o', markersize=4)
        ax9.set_title('📊 周度里程统计', fontsize=12, fontweight='bold')
        ax9.set_ylabel('里程 (km)')
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/running_analysis_report.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ 可视化报告已保存: {output_dir}/running_analysis_report.png")
        
        return fig

# 主函数
if __name__ == '__main__':
    # 注意: 需要将数据保存为 CSV 文件
    # analyzer = RunningAnalyzer('running_data.csv')
    # stats, monthly, activity = analyzer.generate_report()
    # analyzer.create_visualizations()
    
    print("跑步数据分析脚本已准备好！")
    print("使用方法:")
    print("  1. 将跑步数据保存为 running_data.csv")
    print("  2. analyzer = RunningAnalyzer('running_data.csv')")
    print("  3. analyzer.generate_report()")
    print("  4. analyzer.create_visualizations()")
