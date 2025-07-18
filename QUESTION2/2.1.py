import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime
import time
import html
from tqdm import tqdm  # 添加进度条支持


def scrape_weather_data(year, month):
    """
    爬取指定年月的大连历史天气数据

    参数:
        year (int): 年份
        month (int): 月份

    返回:
        list: 包含每日天气数据的字典列表
    """
    url = f"https://www.tianqihoubao.com/lishi/dalian/month/{year}{month:02d}.html"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

    try:
        # 添加请求重试机制
        for retry in range(3):
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()  # 检查请求是否成功
                break
            except (requests.RequestException, requests.Timeout) as e:
                if retry == 2:
                    raise
                time.sleep(5)  # 等待5秒后重试

        # 检测页面编码
        if 'charset' in response.headers.get('content-type', '').lower():
            encoding = re.search(r'charset=([\w-]+)', response.headers['content-type'].lower()).group(1)
        else:
            encoding = 'gb18030'  # 默认编码

        response.encoding = encoding
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')

        if not table:
            print(f"警告: 未找到 {year}-{month:02d} 的数据表格")
            return []

        data = []
        rows = table.find_all('tr')[1:]  # 跳过表头

        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 4:
                continue

            # 提取并清理日期
            date_td = cols[0]
            date_str = ''.join(date_td.stripped_strings)

            if not date_str:
                continue

            # 解析日期
            try:
                date = datetime.strptime(date_str, "%Y年%m月%d日").strftime("%Y-%m-%d")
            except ValueError:
                # 尝试其他日期格式
                nums = re.findall(r'\d+', date_str)
                if len(nums) >= 3:
                    date = f"{nums[0]}-{nums[1].zfill(2)}-{nums[2].zfill(2)}"
                else:
                    print(f"警告: 无法解析日期 '{date_str}'，跳过该记录")
                    continue

            # 提取天气信息
            weather = cols[1].get_text(strip=True).replace('\n', '').replace('\r', '').replace(' ', '')
            temp = cols[2].get_text(strip=True)
            wind = cols[3].get_text(strip=True)

            # 处理天气分割
            weather_parts = [p.strip() for p in weather.split('/')]
            day_weather = weather_parts[0] if weather_parts else ''
            night_weather = weather_parts[1] if len(weather_parts) > 1 else day_weather

            # 提取温度
            temp_match = re.search(r'(-?\d+)℃\s*/\s*(-?\d+)℃', temp)
            if temp_match:
                max_temp = int(temp_match.group(1))
                min_temp = int(temp_match.group(2))
            else:
                max_temp = min_temp = None

            # 处理风力
            wind_parts = [p.strip() for p in wind.split('/')]
            day_wind = wind_parts[0] if wind_parts else ''
            night_wind = wind_parts[1] if len(wind_parts) > 1 else day_wind

            data.append({
                'date': date,
                'day_weather': day_weather,
                'night_weather': night_weather,
                'max_temp': max_temp,
                'min_temp': min_temp,
                'day_wind': day_wind,
                'night_wind': night_wind
            })

        return data

    except Exception as e:
        print(f"错误: 爬取 {year}-{month:02d} 数据失败 - {str(e)}")
        return []


def validate_data(data):
    """
    验证并清理数据

    参数:
        data (list): 原始数据列表

    返回:
        list: 清理后的数据列表
    """
    validated = []
    for record in data:
        # 检查必填字段
        if not record.get('date') or not record.get('day_weather'):
            continue

        # 确保温度在合理范围内
        if record['max_temp'] is not None and (record['max_temp'] < -50 or record['max_temp'] > 50):
            record['max_temp'] = None
        if record['min_temp'] is not None and (record['min_temp'] < -50 or record['min_temp'] > 50):
            record['min_temp'] = None

        validated.append(record)
    return validated


def main():
    # 爬取数据
    all_data = []
    years = range(2022, 2025)  # 2022-2024年
    months = range(1, 13)  # 1-12月

    print("开始爬取大连历史天气数据...")

    for year in tqdm(years, desc="年份进度"):
        for month in tqdm(months, desc=f"{year}年月份进度", leave=False):
            monthly_data = scrape_weather_data(year, month)
            all_data.extend(monthly_data)
            time.sleep(2 + 3 * random.random())  # 随机延迟2-5秒

    # 数据验证和清理
    print("\n数据爬取完成，正在进行验证和清理...")
    all_data = validate_data(all_data)

    if not all_data:
        print("错误: 没有获取到有效数据！")
        return

    # 转换为DataFrame并保存
    weather_df = pd.DataFrame(all_data)

    # 转换日期列为datetime类型并排序
    weather_df['date'] = pd.to_datetime(weather_df['date'], errors='coerce')
    weather_df = weather_df.dropna(subset=['date']).sort_values('date')

    # 数据预览
    print("\n数据预览:")
    print(weather_df.head())
    print("\n数据统计信息:")
    print(weather_df.describe(include='all'))

    # 保存数据
    output_file = 'dalian_weather_2022_2024.csv'
    weather_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n成功保存 {len(weather_df)} 条天气数据到 {output_file}")


if __name__ == "__main__":
    import random

    main()