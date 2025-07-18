import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置中文显示
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


def load_data():
    """加载天气数据"""
    try:
        df = pd.read_csv('dalian_weather_2022_2024.csv', parse_dates=['date'])
        return df
    except FileNotFoundError:
        print("请先运行任务一代码获取数据")
        exit()


def analyze_temperature(df):
    """分析温度数据"""
    # 提取月份
    df['month'] = df['date'].dt.month

    # 计算每月平均温度
    monthly_avg = df.groupby(['month']).agg({
        'max_temp': 'mean',
        'min_temp': 'mean'
    }).reset_index()

    # 保留2位小数
    monthly_avg['max_temp'] = monthly_avg['max_temp'].round(1)
    monthly_avg['min_temp'] = monthly_avg['min_temp'].round(1)

    return monthly_avg


def plot_temperature_trend(monthly_avg):
    """绘制温度变化趋势图"""
    plt.figure(figsize=(12, 6))

    # 绘制最高气温折线
    max_line = plt.plot(monthly_avg['month'], monthly_avg['max_temp'],
                        label='平均最高气温', marker='o', linewidth=2, markersize=8)

    # 绘制最低气温折线
    min_line = plt.plot(monthly_avg['month'], monthly_avg['min_temp'],
                        label='平均最低气温', marker='o', linewidth=2, markersize=8)

    # 在坐标点上添加数字标签
    for x, y in zip(monthly_avg['month'], monthly_avg['max_temp']):
        plt.text(x, y + 0.3, f'{y}℃', ha='center', va='bottom', fontsize=10, color=max_line[0].get_color())

    for x, y in zip(monthly_avg['month'], monthly_avg['min_temp']):
        plt.text(x, y - 0.3, f'{y}℃', ha='center', va='top', fontsize=10, color=min_line[0].get_color())

    # 图表装饰
    plt.title('大连市2022-2024年月平均气温变化趋势', fontsize=16, pad=20)
    plt.xlabel('月份', fontsize=14)
    plt.ylabel('温度(℃)', fontsize=14)
    plt.xticks(range(1, 13), [f'{m}月' for m in range(1, 13)])
    plt.ylim(monthly_avg['min_temp'].min() - 2, monthly_avg['max_temp'].max() + 2)  # 调整y轴范围

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)

    # 保存图表
    plt.tight_layout()
    plt.savefig('dalian_monthly_temp_trend_with_labels.png', dpi=300)
    plt.show()


def main():
    df = load_data()
    monthly_avg = analyze_temperature(df)
    plot_temperature_trend(monthly_avg)


if __name__ == "__main__":
    main()