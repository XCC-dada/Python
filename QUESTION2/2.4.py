import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文显示
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


def classify_weather(weather):
    """天气状况分类"""
    if not isinstance(weather, str):
        return '未知'
    if '晴' in weather:
        return '晴天'
    elif '云' in weather or '昙' in weather:
        return '多云'
    elif '阴' in weather:
        return '阴天'
    elif '雨' in weather:
        return '雨天'
    elif '雪' in weather:
        return '雪天'
    elif '雾' in weather or '霾' in weather:
        return '雾/霾'
    else:
        return '其他'


def load_and_process_data():
    """加载并处理数据"""
    try:
        df = pd.read_csv('dalian_weather_2022_2024.csv', parse_dates=['date'])
        df['month'] = df['date'].dt.month

        # 分类天气状况
        df['day_weather_type'] = df['day_weather'].apply(classify_weather)
        df['night_weather_type'] = df['night_weather'].apply(classify_weather)

        return df
    except FileNotFoundError:
        print("请先运行任务一代码获取数据")
        exit()


def plot_weather_distribution(df):
    """绘制天气分布图"""
    # 统计天气分布
    weather_day = df.groupby(['month', 'day_weather_type']).size().unstack().fillna(0)
    weather_night = df.groupby(['month', 'night_weather_type']).size().unstack().fillna(0)

    # 设置颜色
    colors = plt.cm.Set3(np.linspace(0, 1, 7))

    # 绘制白天天气分布
    plt.figure(figsize=(14, 6))
    weather_day.plot(kind='bar', stacked=True, color=colors, width=0.8)
    plt.title('大连市2022-2024年白天天气状况分布', fontsize=16, pad=20)
    plt.xlabel('月份', fontsize=14)
    plt.ylabel('天数', fontsize=14)
    plt.xticks(range(12), [f'{m}月' for m in range(1, 13)], rotation=0)
    plt.legend(title='天气状况', bbox_to_anchor=(1.05, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('dalian_day_weather_dist.png', dpi=300)
    plt.show()

    # 绘制夜间天气分布
    plt.figure(figsize=(14, 6))
    weather_night.plot(kind='bar', stacked=True, color=colors, width=0.8)
    plt.title('大连市2022-2024年夜间天气状况分布', fontsize=16, pad=20)
    plt.xlabel('月份', fontsize=14)
    plt.ylabel('天数', fontsize=14)
    plt.xticks(range(12), [f'{m}月' for m in range(1, 13)], rotation=0)
    plt.legend(title='天气状况', bbox_to_anchor=(1.05, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('dalian_night_weather_dist.png', dpi=300)
    plt.show()


def main():
    df = load_and_process_data()
    plot_weather_distribution(df)


if __name__ == "__main__":
    main()