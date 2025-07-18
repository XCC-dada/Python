import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
from matplotlib import font_manager
import os

# 创建输出目录
os.makedirs('output', exist_ok=True)


def scrape_expert_data():
    """爬取双色球专家数据"""
    try:
        # 设置Selenium
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # 无头模式
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

        # 访问专家页面
        url = 'https://www.zhcw.com/zj/'
        driver.get(url)
        time.sleep(3)  # 等待页面加载

        # 获取专家数据
        experts = []
        expert_elements = driver.find_elements(By.CSS_SELECTOR, '.expert-list li')

        for elem in expert_elements:
            try:
                expert_id = elem.get_attribute('data-id')
                name = elem.find_element(By.CSS_SELECTOR, '.name').text
                level = elem.find_element(By.CSS_SELECTOR, '.level').text
                experience = elem.find_element(By.CSS_SELECTOR, '.experience').text.replace('彩龄：', '')
                articles = elem.find_element(By.CSS_SELECTOR, '.articles').text.replace('文章：', '').replace('篇', '')
                wins = elem.find_element(By.CSS_SELECTOR, '.wins').text.replace('中奖：', '').replace('次', '')

                experts.append({
                    '专家ID': expert_id,
                    '姓名': name,
                    '双色球专家等级': level,
                    '彩龄(年)': int(experience),
                    '文章数量(篇)': int(articles),
                    '双色球获奖总次数': int(wins)
                })
            except Exception as e:
                print(f"解析专家数据时出错: {e}")
                continue

        # 保存原始数据
        df = pd.DataFrame(experts)
        df.to_csv('raw_expert_data.csv', index=False, encoding='utf_8_sig')
        print("专家数据爬取完成，已保存为 raw_expert_data.csv")

        return df

    except Exception as e:
        print(f"爬取过程中出错: {e}")
        return pd.DataFrame()
    finally:
        driver.quit()


def clean_expert_data(input_path='raw_expert_data.csv', output_path='cleaned_expert_data.csv'):
    """清洗专家数据"""
    try:
        df = pd.read_csv(input_path)

        # 数据清洗
        # 1. 去除无效数据
        df = df.dropna()

        # 2. 计算单位彩龄中奖率
        df['单位彩龄中奖率'] = df['双色球获奖总次数'] / df['彩龄(年)']

        # 3. 保存清洗后的数据
        df.to_csv(output_path, index=False, encoding='utf_8_sig')
        print(f"数据清洗完成，已保存为 {output_path}")

        return df

    except FileNotFoundError:
        print(f"错误：未找到输入文件 {input_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"数据清洗过程中出错: {e}")
        return pd.DataFrame()


def visualize_expert_data(input_path='cleaned_expert_data.csv'):
    """可视化专家数据"""
    try:
        # 设置中文字体
        try:
            font_path = "C:/Windows/Fonts/simhei.ttf"
            font_prop = font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
            plt.rcParams['axes.unicode_minus'] = False
            print("中文字体设置成功。")
        except FileNotFoundError:
            print("警告：未找到simhei.ttf字体，图表中的中文可能无法正确显示。")

        # 读取数据
        df = pd.read_csv(input_path)

        # 1. 彩龄分布
        plt.figure(figsize=(10, 6))
        sns.histplot(df['彩龄(年)'], bins=10, kde=True)
        plt.title('专家彩龄分布')
        plt.xlabel('彩龄 (年)')
        plt.ylabel('专家数量')
        plt.grid(axis='y', alpha=0.75)
        plt.savefig('output/彩龄分布.png')
        plt.show()

        # 2. 文章数量分布
        plt.figure(figsize=(10, 6))
        sns.histplot(df['文章数量(篇)'], bins=20, kde=True)
        plt.title('专家发表文章数量分布')
        plt.xlabel('文章数量 (篇)')
        plt.ylabel('专家数量')
        plt.grid(axis='y', alpha=0.75)
        plt.savefig('output/文章数量分布.png')
        plt.show()

        # 3. 专家等级分布
        plt.figure(figsize=(10, 6))
        level_order = ['无等级', '初级', '中级', '高级', '特级', '天王级']
        sns.countplot(data=df, x='双色球专家等级', order=level_order)
        plt.title('双色球专家等级分布')
        plt.xlabel('专家等级')
        plt.ylabel('专家数量')
        plt.grid(axis='y', alpha=0.75)
        plt.savefig('output/双色球专家等级分布.png')
        plt.show()

        # 4. 彩龄与获奖次数关系
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='彩龄(年)', y='双色球获奖总次数', alpha=0.6)
        sns.regplot(data=df, x='彩龄(年)', y='双色球获奖总次数', scatter=False, color='red')
        plt.title('彩龄与双色球获奖总次数的关系')
        plt.xlabel('彩龄（年）')
        plt.ylabel('双色球获奖总次数')
        plt.grid(True)
        plt.savefig('output/彩龄与获奖关系.png')
        plt.show()

        # 5. 文章数量与获奖次数关系
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='文章数量(篇)', y='双色球获奖总次数', alpha=0.6)
        sns.regplot(data=df, x='文章数量(篇)', y='双色球获奖总次数', scatter=False, color='red')
        plt.title('文章数量与双色球获奖总次数的关系')
        plt.xlabel('文章数量（篇）')
        plt.ylabel('双色球获奖总次数')
        plt.grid(True)
        plt.savefig('output/文章数量与获奖关系.png')
        plt.show()

        # 6. 专家等级与获奖次数关系
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=df, x='双色球专家等级', y='双色球获奖总次数', order=level_order)
        sns.stripplot(data=df, x='双色球专家等级', y='双色球获奖总次数', order=level_order, color=".25", alpha=0.6)
        plt.title('双色球专家等级与获奖总次数的关系')
        plt.xlabel('专家等级')
        plt.ylabel('双色球获奖总次数')
        plt.grid(axis='y', alpha=0.75)
        plt.savefig('output/专家等级与获奖关系.png')
        plt.show()

        # 7. 单位彩龄中奖率分析
        # 文章数量与单位彩龄中奖率
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='文章数量(篇)', y='单位彩龄中奖率', alpha=0.6)
        sns.regplot(data=df, x='文章数量(篇)', y='单位彩龄中奖率', scatter=False, color='red')
        plt.title('文章数量与单位彩龄中奖率的关系')
        plt.xlabel('文章数量（篇）')
        plt.ylabel('单位彩龄中奖率')
        plt.grid(True)
        plt.savefig('output/文章数量与单位彩龄中奖率关系.png')
        plt.show()

        # 彩龄与单位彩龄中奖率
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='彩龄(年)', y='单位彩龄中奖率', alpha=0.6)
        sns.regplot(data=df, x='彩龄(年)', y='单位彩龄中奖率', scatter=False, color='red')
        plt.title('彩龄与单位彩龄中奖率的关系')
        plt.xlabel('彩龄（年）')
        plt.ylabel('单位彩龄中奖率')
        plt.grid(True)
        plt.savefig('output/彩龄与单位彩龄中奖率关系.png')
        plt.show()

        # 专家等级与单位彩龄中奖率
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=df, x='双色球专家等级', y='单位彩龄中奖率', order=level_order)
        sns.stripplot(data=df, x='双色球专家等级', y='单位彩龄中奖率', order=level_order, color=".25", alpha=0.6)
        plt.title('双色球专家等级与单位彩龄中奖率的关系')
        plt.xlabel('专家等级')
        plt.ylabel('单位彩龄中奖率')
        plt.grid(axis='y', alpha=0.75)
        plt.savefig('output/专家等级与单位彩龄中奖率关系.png')
        plt.show()

    except FileNotFoundError:
        print(f"错误：未找到输入文件 {input_path}")
    except Exception as e:
        print(f"可视化过程中出错: {e}")


def analyze_expert_performance(df):
    """分析专家表现"""
    try:
        # 1. 基本统计
        print("=" * 50)
        print("专家基本属性统计:")
        print("=" * 50)
        print(f"专家总数: {len(df)}")
        print(f"平均彩龄: {df['彩龄(年)'].mean():.1f} 年")
        print(f"平均文章数量: {df['文章数量(篇)'].mean():.1f} 篇")
        print(f"平均获奖次数: {df['双色球获奖总次数'].mean():.1f} 次")
        print(f"平均单位彩龄中奖率: {df['单位彩龄中奖率'].mean():.2f} 次/年")

        # 2. 按等级分组统计
        print("\n" + "=" * 50)
        print("按专家等级分组统计:")
        print("=" * 50)
        level_stats = df.groupby('双色球专家等级').agg({
            '彩龄(年)': ['count', 'mean', 'median', 'std'],
            '文章数量(篇)': ['mean', 'median', 'std'],
            '双色球获奖总次数': ['mean', 'median', 'std'],
            '单位彩龄中奖率': ['mean', 'median', 'std']
        })
        print(level_stats)

        # 3. 相关性分析
        print("\n" + "=" * 50)
        print("属性间相关性分析:")
        print("=" * 50)
        corr_matrix = df[['彩龄(年)', '文章数量(篇)', '双色球获奖总次数', '单位彩龄中奖率']].corr()
        print(corr_matrix)

    except Exception as e:
        print(f"分析过程中出错: {e}")


if __name__ == "__main__":
    # 1. 爬取数据
    print("开始爬取专家数据...")
    expert_df = scrape_expert_data()

    if not expert_df.empty:
        # 2. 清洗数据
        print("\n开始清洗专家数据...")
        cleaned_df = clean_expert_data()

        if not cleaned_df.empty:
            # 3. 可视化数据
            print("\n开始可视化专家数据...")
            visualize_expert_data()

            # 4. 分析专家表现
            print("\n开始分析专家表现...")
            analyze_expert_performance(cleaned_df)

            print("\n所有任务完成!")
        else:
            print("数据清洗失败，无法继续后续步骤")
    else:
        print("数据爬取失败，无法继续后续步骤")