# task2_number_analysis_fixed.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import random
import ast
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    """加载截至2025-07-01前100期数据"""
    if not os.path.exists('dlt_last_100_before_20250701.csv'):
        raise FileNotFoundError("请先运行任务1获取数据文件")

    df = pd.read_csv('dlt_last_100_before_20250701.csv')
    # 安全转换字符串格式的列表
    df['前区'] = df['前区'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['后区'] = df['后区'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df


def analyze_frequency(df):
    """统计号码出现频率"""
    front_numbers = [num for sublist in df['前区'] for num in sublist]
    back_numbers = [num for sublist in df['后区'] for num in sublist]
    return Counter(front_numbers), Counter(back_numbers)


def plot_frequency(front_counts, back_counts):
    """绘制频率分布图"""
    # 前区（1-35）
    front_df = pd.DataFrame({
        '号码': [f"{int(num):02d}" for num in range(1, 36)],
        '出现次数': [front_counts.get(str(num), 0) for num in range(1, 36)]
    })

    plt.figure(figsize=(18, 6))
    sns.barplot(data=front_df, x='号码', y='出现次数', color='firebrick')
    plt.title('前区号码出现频率（1-35）', fontsize=14, pad=20)
    plt.xlabel('')
    plt.ylabel('出现次数', labelpad=10)
    plt.xticks(rotation=90)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('front_number_frequency.png', dpi=300, bbox_inches='tight')

    # 后区（1-12）
    back_df = pd.DataFrame({
        '号码': [f"{int(num):02d}" for num in range(1, 13)],
        '出现次数': [back_counts.get(str(num), 0) for num in range(1, 13)]
    })

    plt.figure(figsize=(12, 5))
    sns.barplot(data=back_df, x='号码', y='出现次数', color='steelblue')
    plt.title('后区号码出现频率（1-12）', fontsize=14, pad=20)
    plt.xlabel('')
    plt.ylabel('出现次数', labelpad=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('back_number_frequency.png', dpi=300, bbox_inches='tight')


def predict_numbers(front_counts, back_counts):
    """智能号码预测"""
    # 定义热号（>平均频率1.2倍）、冷号（<0.8倍）
    front_avg = sum(front_counts.values()) / 35
    hot_front = [k for k, v in front_counts.items() if v > front_avg * 1.2]
    cold_front = [k for k, v in front_counts.items() if v < front_avg * 0.8]
    mid_front = [k for k, v in front_counts.items() if front_avg * 0.8 <= v <= front_avg * 1.2]

    back_avg = sum(back_counts.values()) / 12
    hot_back = [k for k, v in back_counts.items() if v > back_avg * 1.2]
    cold_back = [k for k, v in back_counts.items() if v < back_avg * 0.8]

    # 组合策略：3热号 + 1温号 + 1冷号（前区） | 1热 + 1冷（后区）
    selection = {
        '前区': random.sample(hot_front, min(3, len(hot_front))) +
                random.sample(mid_front, min(1, len(mid_front))) +
                random.sample(cold_front, min(1, len(cold_front))),
        '后区': random.sample(hot_back, 1) + random.sample(cold_back, 1)
    }

    # 格式化输出
    print("\n=== 智能预测结果 ===")
    print(f"前区推荐: {' '.join(sorted(selection['前区'], key=int))}")
    print(f"后区推荐: {' '.join(sorted(selection['后区'], key=int))}")
    print("\n策略说明：")
    print("- 前区组合：3个热号 + 1个温号 + 1个冷号")
    print("- 后区组合：1个热号 + 1个冷号")

    return selection


def main():
    print("=== 大乐透号码分析系统（基于2025-07-01前100期数据） ===")

    try:
        df = load_data()
        print(f"数据加载成功，共{len(df)}期开奖记录")

        front_counts, back_counts = analyze_frequency(df)

        print("\n正在生成可视化图表...")
        plot_frequency(front_counts, back_counts)

        print("\n正在进行智能选号...")
        predict_numbers(front_counts, back_counts)

        print("\n分析完成！结果已保存至：")
        print("- front_number_frequency.png")
        print("- back_number_frequency.png")

    except Exception as e:
        print(f"\n错误：{str(e)}")
        print("建议检查：")
        print("1. 是否已运行任务1生成数据文件")
        print("2. 数据文件是否完整")


if __name__ == "__main__":
    main()