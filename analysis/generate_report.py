#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跑步数据分析报告生成器
自动分析并生成详细的训练数据报告
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from running_analyzer import RunningAnalyzer

def generate_detailed_report(analyzer):
    """生成详细的文本报告"""
    
    report = []
    report.append("\n" + "="*80)
    report.append("🏃 跑步训练数据分析详细报告")
    report.append("="*80)
    
    # 基础统计
    stats = analyzer.get_statistics()
    report.append("\n📊 基础统计信息")
    report.append("-" * 80)
    report.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"数据覆盖期: {analyzer.df['日期'].min().date()} 至 {analyzer.df['日期'].max().date()}")
    report.append(f"总训练天数: {(analyzer.df['日期'].max() - analyzer.df['日期'].min()).days} 天")
    report.append(f"")
    report.append(f"总训练次数: {int(stats['总记录数'])} 次")
    report.append(f"总里程: {stats['总里程(km)']:.2f} km")
    report.append(f"总训练时间: {stats['总时间(小时)']:.2f} 小时 ({int(stats['总时间(小时)']/24)} 天 {int(stats['总时间(小时)']%24)} 小时)")
    report.append(f"平均单次距离: {stats['平均单次距离(km)']:.2f} km")
    report.append(f"总热量消耗: {stats['总热量消耗(kcal)']:.0f} kcal")
    
    # 配速分析
    report.append("\n⚡ 配速分析")
    report.append("-" * 80)
    report.append(f"最快配速: {stats['最佳配速(min/km)']:.2f} min/km")
    report.append(f"平均配速: {stats['平均配速(min/km)']:.2f} min/km")
    
    # 距离分析
    report.append("\n📏 距离分析")
    report.append("-" * 80)
    report.append(f"最长单次距离: {stats['最长单次距离(km)']:.2f} km")
    report.append(f"中位数距离: {analyzer.df['距离'].median():.2f} km")
    report.append(f"距离标准差: {analyzer.df['距离'].std():.2f} km")
    
    distances = analyzer.df['距离'].values
    report.append(f"短距离 (<5km) 训练: {len([d for d in distances if d < 5])} 次")
    report.append(f"中距离 (5-10km) 训练: {len([d for d in distances if 5 <= d < 10])} 次")
    report.append(f"长距离 (10-15km) 训练: {len([d for d in distances if 10 <= d < 15])} 次")
    report.append(f"超长距离 (15km+) 训练: {len([d for d in distances if d >= 15])} 次")
    
    # 心率分析
    report.append("\n💓 心率分析")
    report.append("-" * 80)
    report.append(f"平均心率: {stats['平均心率(bpm)']:.0f} bpm")
    report.append(f"最大心率: {stats['最大心率(bpm)']:.0f} bpm")
    report.append(f"最小心率: {analyzer.df['平均心率'].min():.0f} bpm")
    report.append(f"心率标准差: {analyzer.df['平均心率'].std():.0f} bpm")
    
    # 月度分析
    report.append("\n📅 月度分析")
    report.append("-" * 80)
    monthly = analyzer.get_monthly_stats()
    for month, row in monthly.iterrows():
        report.append(f"{month}: {row['距离']:.0f}km / {int(row['次数'])} 次 / {row['平均配速']:.2f}min/km平均配速")
    
    # 活动类型分析
    report.append("\n🏷️ 活动类型分析")
    report.append("-" * 80)
    activity = analyzer.get_activity_type_stats()
    for activity_type, row in activity.iterrows():
        pct = (row['距离'] / stats['总里程(km)']) * 100
        report.append(f"{activity_type}: {row['距离']:.0f}km ({pct:.1f}%) / {int(row['次数'])} 次")
    
    # 个人纪录
    report.append("\n🎯 个人纪录 (PB)")
    report.append("-" * 80)
    
    max_dist = analyzer.df.loc[analyzer.df['距离'].idxmax()]
    report.append(f"最长距离: {max_dist['距离']:.2f}km on {max_dist['日期'].date()} ({max_dist['标题']})")
    
    max_cal = analyzer.df.loc[analyzer.df['热量消耗'].idxmax()]
    report.append(f"最多热量: {max_cal['热量消耗']:.0f}kcal on {max_cal['日期'].date()}")
    
    best_pace = analyzer.df.loc[analyzer.df['平均配速_秒'].idxmin()]
    report.append(f"最快配速: {best_pace['平均速度']} min/km on {best_pace['日期'].date()}")
    
    # 训练频率
    report.append("\n📈 训练频率")
    report.append("-" * 80)
    total_days = (analyzer.df['日期'].max() - analyzer.df['日期'].min()).days + 1
    weekly_avg = len(analyzer.df) / (total_days / 7)
    report.append(f"平均每周训练次数: {weekly_avg:.1f} 次")
    report.append(f"平均每周里程: {stats['总里程(km)'] / (total_days / 7):.1f} km")
    
    # 最活跃的日期
    analyzer.df['日期_仅日期'] = analyzer.df['日期'].dt.date
    daily_counts = analyzer.df.groupby('日期_仅日期').size()
    max_day = daily_counts.idxmax()
    report.append(f"最活跃日期: {max_day} ({daily_counts[max_day]} 次训练)")
    
    report_text = "\n".join(report)
    print(report_text)
    
    # 保存报告
    with open('running_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    return report_text

if __name__ == '__main__':
    print("🏃 正在加载数据...")
    analyzer = RunningAnalyzer('data/running_data.tsv')
    
    print("📊 正在生成报告...")
    generate_detailed_report(analyzer)
    
    print("\n📈 正在生成图表...")
    analyzer.create_visualizations('output')
    
    print("✅ 分析完成！")
    print("\n生成的文件:")
    print("  - running_analysis_report.txt (详细文本报告)")
    print("  - output/running_analysis_report.png (可视化图表)")
