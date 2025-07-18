# task3_weekday_analysis_final.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
import ast
import os
from datetime import datetime
import matplotlib.font_manager as fm


# 1. 解决中文显示问题（所有操作系统通用方案）
def set_chinese_font():
    """设置中文字体，自动选择可用字体"""
    font_paths = [
        # Windows字体
        'C:/Windows/Fonts/simhei.ttf',  # 黑体
        'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
        # Mac字体
        '/System/Library/Fonts/STHeiti Medium.ttc',  # 华文黑体
        # Linux字体
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font_prop = fm.FontProperties(fname=font_path)
                plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"已设置中文字体: {font_prop.get_name()}")
                return
            except:
                continue

    # 如果上述字体都不可用，尝试系统默认字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        try:
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            print("警告: 未能找到合适的中文字体，图表可能显示方框")


# 设置中文字体
set_chinese_font()

# 2. 设置图表样式
sns.set_style("whitegrid", {'font.sans-serif': plt.rcParams['font.sans-serif']})
plt.style.use('ggplot')


def load_and_preprocess():
    """加载并预处理数据"""
    if not os.path.exists('dlt_last_100_before_20250701.csv'):
        raise FileNotFoundError("请先运行任务1生成数据文件")

    df = pd.read_csv('dlt_last_100_before_20250701.csv')

    # 转换数据类型
    df['开奖日期'] = pd.to_datetime(df['开奖日期'])
    df['前区'] = df['前区'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['后区'] = df['后区'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # 添加星期几信息（0-6对应周一到周日）
    df['星期'] = df['开奖日期'].dt.dayofweek
    df['开奖日'] = df['星期'].map({0: '周一', 2: '周三', 5: '周六'})

    # 仅保留大乐透开奖日（周一、三、六）
    df = df[df['开奖日'].notna()]
    return df


def plot_sales_comparison(df):
    """绘制不同开奖日的销售额对比"""
    plt.figure(figsize=(12, 6))

    # 使用更现代的violinplot
    ax = sns.violinplot(data=df, x='开奖日', y='销售额', order=['周一', '周三', '周六'],
                        inner="quartile", palette="Set2")

    # 添加统计标注
    medians = df.groupby('开奖日')['销售额'].median()
    for i, day in enumerate(['周一', '周三', '周六']):
        ax.text(i, medians[day] + 3e6, f'中位数:\n{medians[day] / 1e8:.2f}亿',
                ha='center', va='bottom', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.title('不同开奖日的销售额分布对比（2025-07-01前100期）', pad=20, fontsize=14)
    plt.xlabel('')
    plt.ylabel('销售额（元）', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # 确保保存图片时也包含中文
    plt.savefig('sales_by_weekday.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_number_heatmaps(df):
    """绘制按开奖日的号码热力图"""
    # 初始化计数矩阵
    front_counts = np.zeros((3, 35))  # 3个开奖日 x 35个前区号码
    back_counts = np.zeros((3, 12))  # 3个开奖日 x 12个后区号码

    # 映射星期到索引
    day_to_idx = {'周一': 0, '周三': 1, '周六': 2}

    # 统计出现次数
    for _, row in df.iterrows():
        day_idx = day_to_idx[row['开奖日']]
        for num in row['前区']:
            front_counts[day_idx, int(num) - 1] += 1
        for num in row['后区']:
            back_counts[day_idx, int(num) - 1] += 1

    # 前区热力图
    plt.figure(figsize=(18, 5))
    sns.heatmap(front_counts,
                xticklabels=[f"{i + 1:02d}" for i in range(35)],
                yticklabels=['周一', '周三', '周六'],
                cmap="YlOrRd", annot=True, fmt=".0f",
                cbar_kws={'label': '出现次数'},
                linewidths=0.5, linecolor='lightgray')
    plt.title('前区号码在不同开奖日的出现次数', pad=15, fontsize=14)
    plt.xlabel('前区号码', fontsize=12)
    plt.ylabel('开奖日', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig('front_heatmap_by_weekday.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 后区热力图
    plt.figure(figsize=(12, 4))
    sns.heatmap(back_counts,
                xticklabels=[f"{i + 1:02d}" for i in range(12)],
                yticklabels=['周一', '周三', '周六'],
                cmap="Blues", annot=True, fmt=".0f",
                cbar_kws={'label': '出现次数'},
                linewidths=0.5, linecolor='lightgray')
    plt.title('后区号码在不同开奖日的出现次数', pad=15, fontsize=14)
    plt.xlabel('后区号码', fontsize=12)
    plt.ylabel('开奖日', fontsize=12)
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig('back_heatmap_by_weekday.png', dpi=300, bbox_inches='tight')
    plt.show()


def statistical_test(df):
    """执行统计检验"""
    # 准备数据
    mon = df[df['开奖日'] == '周一']['销售额']
    wed = df[df['开奖日'] == '周三']['销售额']
    sat = df[df['开奖日'] == '周六']['销售额']

    # Kruskal-Wallis检验（非参数方差分析）
    h_stat, p_value = kruskal(mon, wed, sat)

    # 计算描述性统计
    stats_df = df.groupby('开奖日')['销售额'].agg(['mean', 'median', 'std', 'count'])
    stats_df['mean'] = stats_df['mean'] / 1e8  # 转换为亿单位
    stats_df['median'] = stats_df['median'] / 1e8

    print("\n=== 统计检验结果 ===")
    print("开奖日 | 平均销售额(亿) | 中位数(亿) | 标准差 | 期数")
    print("------|--------------|-----------|-------|----")
    for day in ['周一', '周三', '周六']:
        row = stats_df.loc[day]
        print(f"{day} | {row['mean']:.2f} | {row['median']:.2f} | {row['std'] / 1e6:.1f}百万 | {row['count']}")

    print(f"\nKruskal-Wallis检验统计量: {h_stat:.3f}")
    print(f"P值: {p_value:.4f}")

    if p_value < 0.05:
        max_day = stats_df['mean'].idxmax()
        print(f"\n结论: 不同开奖日的销售额存在显著差异 (p < 0.05)")
        print(f"销售额最高的开奖日是: {max_day}（平均{stats_df.loc[max_day, 'mean']:.2f}亿）")
    else:
        print("\n结论: 未发现不同开奖日的销售额有显著差异")


def main():
    print("=== 任务3：开奖日分析 ===")

    try:
        # 数据加载
        df = load_and_preprocess()
        print(f"数据加载成功，共分析{len(df)}期开奖记录")
        print("\n开奖日分布:")
        print(df['开奖日'].value_counts().sort_index().to_string())

        # 销售额对比
        print("\n正在生成销售额对比图...")
        plot_sales_comparison(df)

        # 号码分布热力图
        print("正在生成号码热力图...")
        plot_number_heatmaps(df)

        # 统计检验
        print("正在进行统计检验...")
        statistical_test(df)

        print("\n分析完成！生成图表：")
        print("- sales_by_weekday.png")
        print("- front_heatmap_by_weekday.png")
        print("- back_heatmap_by_weekday.png")

    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("请检查：")
        print("1. 是否已运行任务1生成数据文件")
        print("2. 数据文件格式是否正确")
        print("3. 如果中文仍显示为方框，请尝试以下解决方案：")
        print("   a. 安装中文字体（如微软雅黑）到系统字体目录")
        print("   b. 在代码中手动指定字体路径（修改set_chinese_font()函数）")


if __name__ == "__main__":
    # 初始化字体设置
    main()