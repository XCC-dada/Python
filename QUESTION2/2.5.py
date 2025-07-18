# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.font_manager as fm
import warnings

warnings.filterwarnings("ignore")


# 1. 设置中文字体
def set_chinese_font():
    try:
        # 尝试使用系统自带字体
        font_path = None
        for font in fm.fontManager.ttflist:
            if 'SimHei' in font.name:
                font_path = font.fname
                break

        if font_path is None:
            for font in fm.fontManager.ttflist:
                if 'Microsoft YaHei' in font.name:
                    font_path = font.fname
                    break

        if font_path:
            plt.rcParams['font.sans-serif'] = [fm.FontProperties(fname=font_path).get_name()]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"已设置中文字体: {fm.FontProperties(fname=font_path).get_name()}")
        else:
            print("警告: 未找到合适的中文字体，图表可能显示异常")
    except Exception as e:
        print(f"字体设置出错: {str(e)}")


# 2. 生成模拟数据函数
def generate_sample_data():
    """生成包含日期和温度的模拟数据"""
    dates = pd.date_range('2020-01-01', '2025-06-30')
    temps = 10 + 15 * np.sin(2 * np.pi * (dates.dayofyear / 365)) + np.random.normal(0, 3, len(dates))
    return pd.DataFrame({'日期': dates, '温度': temps})


# 3. 主分析函数
def analyze_temperature():
    # 设置中文字体
    set_chinese_font()

    # 获取数据
    df = generate_sample_data()
    df['日期'] = pd.to_datetime(df['日期'])
    df.set_index('日期', inplace=True)

    # 划分数据集
    train = df.loc['2020-01-01':'2024-12-31', '温度']
    test = df.loc['2025-01-01':'2025-06-30', '温度']

    # 转换为月平均
    train_monthly = train.resample('M').mean()
    test_monthly = test.resample('M').mean()

    # 设置SARIMA参数
    order = (1, 0, 1)  # (p,d,q)
    seasonal_order = (1, 1, 0, 12)  # (P,D,Q,m)

    # 训练模型
    model = SARIMAX(train_monthly,
                    order=order,
                    seasonal_order=seasonal_order).fit(disp=False)

    # 预测
    forecast = model.get_forecast(steps=6)
    pred = forecast.predicted_mean
    ci = forecast.conf_int()

    # 可视化设置
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # 绘制数据
    train_line, = ax.plot(train_monthly[-12:].index,
                          train_monthly[-12:].values,
                          'o-', label='历史月平均温度')

    test_line, = ax.plot(test_monthly.index,
                         test_monthly.values,
                         'o-', color='green', label='真实月平均温度')

    pred_line, = ax.plot(test_monthly.index,
                         pred.values,
                         '--x', color='red', label='预测值')

    # 填充置信区间
    ax.fill_between(test_monthly.index,
                    ci.iloc[:, 0],
                    ci.iloc[:, 1],
                    color='gray', alpha=0.2, label='95%置信区间')

    # 设置标题和标签
    plt.title('大连市月平均最高温度预测 (2025年1-6月)', fontsize=14, pad=20)
    plt.xlabel('日期', fontsize=12, labelpad=10)
    plt.ylabel('温度 (℃)', fontsize=12, labelpad=10)

    # 设置坐标轴格式
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.yticks(np.arange(0, 30, 2))

    # 添加坐标值标签
    for x, y in zip(train_monthly[-12:].index, train_monthly[-12:].values):
        ax.text(x, y, f'{y:.1f}', ha='center', va='bottom', fontsize=8)

    for x, y in zip(test_monthly.index, test_monthly.values):
        ax.text(x, y, f'{y:.1f}', ha='center', va='bottom', fontsize=8, color='green')

    for x, y in zip(test_monthly.index, pred.values):
        ax.text(x, y, f'{y:.1f}', ha='center', va='top', fontsize=8, color='red')

    # 添加网格和图示
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left')

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig('temperature_prediction_with_labels.png', dpi=300, bbox_inches='tight')
    print("图表已保存为 temperature_prediction_with_labels.png")
    plt.show()


if __name__ == "__main__":
    analyze_temperature()