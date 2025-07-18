import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文显示
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


def classify_wind(wind_str):
    """风力等级分类"""
    if not isinstance(wind_str, str):
        return '未知'
    if '1-2级' in wind_str:
        return '1-2级'
    elif '3-4级' in wind_str:
        return '3-4级'
    elif '5-6级' in wind_str:
        return '5-6级'
    elif '7-8级' in wind_str:
        return '7-8级'
    elif '9-10级' in wind_str:
        return '9-10级'
    else:
        return '其他'


def load_and_process_data():
    """加载并处理数据"""
    try:
        df = pd.read_csv('dalian_weather_2022_2024.csv', parse_dates=['date'])
        df['month'] = df['date'].dt.month

        # 分类风力等级
        df['day_wind_level'] = df['day_wind'].apply(classify_wind)
        df['night_wind_level'] = df['night_wind'].apply(classify_wind)

        return df
    except FileNotFoundError:
        print("请先运行任务一代码获取数据")
        exit()


def plot_wind_distribution(df):
    """绘制风力分布图"""
    # 统计风力分布
    wind_day = df.groupby(['month', 'day_wind_level']).size().unstack().fillna(0)
    wind_night = df.groupby(['month', 'night_wind_level']).size().unstack().fillna(0)

    # 设置颜色
    colors = plt.cm.Pastel1(np.linspace(0, 1, 6))

    # 绘制白天风力分布
    plt.figure(figsize=(14, 6))
    wind_day.plot(kind='bar', stacked=True, color=colors, width=0.8)
    plt.title('大连市2022-2024年白天风力等级分布', fontsize=16, pad=20)
    plt.xlabel('月份', fontsize=14)
    plt.ylabel('天数', fontsize=14)
    plt.xticks(range(12), [f'{m}月' for m in range(1, 13)], rotation=0)
    plt.legend(title='风力等级', bbox_to_anchor=(1.05, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('dalian_day_wind_dist.png', dpi=300)
    plt.show()

    # 绘制夜间风力分布
    plt.figure(figsize=(14, 6))
    wind_night.plot(kind='bar', stacked=True, color=colors, width=0.8)
    plt.title('大连市2022-2024年夜间风力等级分布', fontsize=16, pad=20)
    plt.xlabel('月份', fontsize=14)
    plt.ylabel('天数', fontsize=14)
    plt.xticks(range(12), [f'{m}月' for m in range(1, 13)], rotation=0)
    plt.legend(title='风力等级', bbox_to_anchor=(1.05, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('dalian_night_wind_dist.png', dpi=300)
    plt.show()


def main():
    df = load_and_process_data()
    plot_wind_distribution(df)


if __name__ == "__main__":
    main()