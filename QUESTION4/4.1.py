# task1_sales_prediction_fixed.py
import requests
import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
plt.rcParams['axes.unicode_minus'] = False


def fetch_dlt_history(end_date='2025-07-01', num_periods=100):
    """
    爬取截至指定日期（默认2025-07-01）之前100期的大乐透数据
    """
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15'
    ]

    api_url = "https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry"
    all_data = []
    current_page = 1
    total_pages = 3  # 预估需要3页数据（每页50期）

    while len(all_data) < num_periods and current_page <= total_pages:
        params = {
            "gameNo": "85",  # 大乐透游戏编号
            "provinceId": "0",
            "pageSize": "50",  # 每页50期
            "isVerify": "1",
            "pageNo": str(current_page)
        }

        headers = {
            'User-Agent': random.choice(USER_AGENTS),
            "Referer": 'https://www.lottery.gov.cn/'
        }

        try:
            response = requests.get(api_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('success'):
                for item in data['value']['list']:
                    try:
                        draw_date = datetime.strptime(item['lotteryDrawTime'], '%Y-%m-%d')
                        if draw_date >= datetime.strptime(end_date, '%Y-%m-%d'):
                            continue  # 跳过7月1日及之后的期数

                        all_data.append({
                            '期号': item['lotteryDrawNum'],
                            '开奖日期': draw_date,
                            '前区': item['lotteryDrawResult'].split()[:5],
                            '后区': item['lotteryDrawResult'].split()[5:7],
                            '销售额': int(item['totalSaleAmount'].replace(',', '')) * 10000  # 转换为元
                        })

                        if len(all_data) >= num_periods:
                            break
                    except Exception as e:
                        print(f"解析数据时出错（期号{item.get('lotteryDrawNum')}）: {e}")
                        continue
        except Exception as e:
            print(f"请求第{current_page}页失败: {e}")
            time.sleep(5)  # 失败后等待5秒再重试
            continue

        current_page += 1
        time.sleep(1)  # 避免请求过快

    if not all_data:
        raise ValueError("未能获取任何数据，请检查网络或API变更")

    df = pd.DataFrame(all_data)
    df = df.sort_values('开奖日期', ascending=False).head(num_periods)  # 确保正好100期
    df.to_csv('dlt_last_100_before_20250701.csv', index=False, encoding='utf_8_sig')
    print(f"成功爬取{len(df)}期数据（截至{end_date}前100期）")
    return df


def predict_sales():
    """ 使用SARIMA模型预测下一期销售额 """
    print("=== 大乐透数据爬取与销售额预测（截至2025-07-01前100期） ===")

    if not os.path.exists('dlt_last_100_before_20250701.csv'):
        print("正在爬取数据...")
        try:
            df = fetch_dlt_history()
        except Exception as e:
            print(f"数据爬取失败: {e}")
            return
    else:
        df = pd.read_csv('dlt_last_100_before_20250701.csv')
        df['开奖日期'] = pd.to_datetime(df['开奖日期'])

    # 按时间排序并提取销售额序列
    df = df.sort_values('开奖日期')
    sales_series = df.set_index('开奖日期')['销售额']

    # 使用SARIMA(1,1,1)(1,1,1,3)模型
    try:
        print("正在训练预测模型...")
        model = SARIMAX(
            sales_series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 3),  # 每周3次开奖
            enforce_stationarity=False
        )
        results = model.fit(disp=False)

        # 预测下一期（2025-07-02）
        forecast = results.get_forecast(steps=1)
        forecast_mean = forecast.predicted_mean.iloc[0]
        conf_int = forecast.conf_int().iloc[0]

        print("\n=== 预测结果 ===")
        print(f"模型参数: SARIMA(1,1,1)(1,1,1,3)")
        print(f"下一期预测销售额: {forecast_mean:,.2f}元")
        print(f"95%置信区间: [{conf_int[0]:,.2f}, {conf_int[1]:,.2f}]")

        # 可视化
        plt.figure(figsize=(12, 6))
        plt.plot(sales_series.index, sales_series, 'b-', label='历史销售额')
        next_date = sales_series.index[-1] + timedelta(days=3)  # 假设下期在3天后
        plt.axhline(y=forecast_mean, color='r', linestyle='--', label='预测值')
        plt.fill_between(
            [sales_series.index[-1], next_date],
            conf_int[0], conf_int[1], color='r', alpha=0.1
        )
        plt.title('大乐透销售额趋势与预测（截至2025-07-01前100期）')
        plt.xlabel('日期')
        plt.ylabel('销售额（元）')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('sales_trend_prediction.png', dpi=300)
        plt.show()

    except Exception as e:
        print(f"模型训练失败: {e}")
        # 备用方案：使用移动平均
        avg_sales = sales_series.rolling(window=5).mean().iloc[-1]
        print(f"\n使用备选方案预测: 近5期平均销售额 {avg_sales:,.2f}元")


if __name__ == "__main__":
    predict_sales()