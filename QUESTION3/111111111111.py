import requests
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from sklearn.linear_model import LinearRegression
import matplotlib
import numpy as np
import os
import time

# 设置matplotlib支持中文
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 会议配置
CONFERENCES = {
    "AAAI": "aaai",
    "IJCAI": "ijcai",
    "CVPR": "cvpr",
    "NeurIPS": "nips",
    "ICML": "icml"
}
START_YEAR = 2020
END_YEAR = 2024  # 不包含2025年


def parse_authors(authors):
    """解析作者信息，处理不同格式"""
    if isinstance(authors, dict):
        author = authors.get('author', [])
        if isinstance(author, list):
            names = []
            for a in author:
                if isinstance(a, dict):
                    names.append(a.get('text', ''))
                else:
                    names.append(str(a))
            return ', '.join(names)
        elif isinstance(author, dict):
            return author.get('text', '')
        else:
            return str(author)
    elif isinstance(authors, list):
        return ', '.join([a.get('text', '') if isinstance(a, dict) else str(a) for a in authors])
    else:
        return str(authors)


def fetch_dblp_api_with_pagination(conf_key, conf_name):
    """从DBLP API爬取会议论文数据，使用分页获取更多数据"""
    papers = []
    for year in range(START_YEAR, END_YEAR + 1):
        year_papers = []
        offset = 0
        max_results = 1000

        while True:
            url = f"https://dblp.org/search/publ/api?q=stream%3Aconf%2F{conf_key}%3A{year}%3A&h={max_results}&f={offset}&format=json"
            try:
                resp = requests.get(url, timeout=15)
                data = resp.json()
                hits = data.get('result', {}).get('hits', {}).get('hit', [])

                if not hits:
                    break

                for hit in hits:
                    info = hit.get('info', {})
                    title = info.get('title', '')
                    authors = parse_authors(info.get('authors', {}))
                    link = info.get('url', '')

                    if title:  # 只保存有标题的论文
                        year_papers.append({
                            "title": title,
                            "authors": authors,
                            "year": year,
                            "conference": conf_name,
                            "link": link
                        })

                # 如果返回的结果少于请求的数量，说明已经获取完所有数据
                if len(hits) < max_results:
                    break

                offset += max_results
                time.sleep(0.5)  # 避免请求过快

            except Exception as e:
                print(f"{conf_name} {year} 年获取失败: {e}")
                break

        papers.extend(year_papers)
        print(f"{conf_name} {year} 年论文数：{len(year_papers)}")

    return papers


def get_all_papers():
    """获取所有会议的论文数据"""
    all_papers = []
    for conf_name, conf_key in CONFERENCES.items():
        print(f"\n正在爬取 {conf_name} ...")
        papers = fetch_dblp_api_with_pagination(conf_key, conf_name)
        print(f"{conf_name} 总论文数：{len(papers)}")
        all_papers.extend(papers)

    df = pd.DataFrame(all_papers)
    print(f"\n总论文数：{len(df)}")
    print(f"DataFrame 列名：{df.columns.tolist()}")

    # 显示每年每会的论文数量统计
    if not df.empty:
        print("\n各会议每年论文数量统计：")
        pivot_table = df.pivot_table(index='year', columns='conference', values='title', aggfunc='count', fill_value=0)
        print(pivot_table)

    return df


def plot_trend(df):
    """绘制各会议论文数量趋势图"""
    if 'conference' not in df.columns or df.empty:
        print("没有爬取到论文数据，无法绘图。")
        return

    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, conf in enumerate(df['conference'].unique()):
        sub = df[df['conference'] == conf]
        count = sub.groupby('year').size()

        if len(count) > 0:
            plt.plot(count.index, count.values, marker='o', label=conf,
                     linewidth=2, markersize=8, color=colors[i % len(colors)])

    plt.xlabel('年份', fontsize=12)
    plt.ylabel('论文数量', fontsize=12)
    plt.title('各会议每年论文数量变化趋势 (2020-2024)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('trend.png', dpi=300, bbox_inches='tight')
    plt.show()


def extract_keywords(titles):
    """提取论文标题关键词，过滤停用词"""
    # 严格的停用词列表，确保过滤掉所有无用词
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'from',
        'into', 'during', 'including', 'until', 'against', 'among', 'throughout', 'despite',
        'towards', 'upon', 'concerning', 'like', 'through', 'within', 'without', 'against',
        'between', 'about', 'over', 'under', 'since', 'before', 'after', 'above', 'below',
        'up', 'down', 'out', 'off', 'on', 'over', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 'you', 'your', 'yours', 'yourself', 'yourselves', 'i',
        'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'what', 'which', 'who', 'whom',
        'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'being', 'been',
        'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'he', 'him',
        'his', 'himself', 'she', 'her', 'hers', 'herself', 'as', 'if', 'or', 'because', 'as',
        'until', 'while', 'of', 'at', 'by', 'for', 'with', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down', 'in', 'out', 'on',
        'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
        'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        'can', 'will', 'just', 'should', 'now', 'also', 'well', 'very', 'even', 'much', 'many',
        'still', 'yet', 'already', 'always', 'never', 'often', 'sometimes', 'usually', 'generally',
        'particularly', 'especially', 'mainly', 'primarily', 'largely', 'mostly', 'chiefly',
        'principally', 'essentially', 'basically', 'fundamentally', 'primarily', 'mainly',
        'largely', 'mostly', 'chiefly', 'principally', 'essentially', 'basically', 'fundamentally',
        'using', 'based', 'towards', 'via', 'using', 'based', 'towards', 'via', 'using', 'based'
    }

    words = []
    for title in titles:
        # 更严格的清理和分词
        title_words = []
        for w in title.split():
            # 清理单词
            cleaned_word = w.lower().strip('.,!?;:()[]{}"\'-')
            # 确保是纯字母且长度大于2且不在停用词中
            if (len(cleaned_word) > 2 and
                    cleaned_word.isalpha() and
                    cleaned_word not in stop_words):
                title_words.append(cleaned_word)
        words.extend(title_words)

    return words


def plot_yearly_wordclouds(df):
    """为每年生成独立的词云图"""
    if df.empty:
        print("没有数据，无法生成词云。")
        return

    # 创建输出目录
    os.makedirs('wordclouds', exist_ok=True)

    # 确保每年都有数据
    for year in range(START_YEAR, END_YEAR + 1):
        year_data = df[df['year'] == year]
        if year_data.empty:
            print(f"{year}年没有数据，跳过词云生成")
            continue

        # 提取该年的关键词
        titles = year_data['title'].tolist()
        words = extract_keywords(titles)

        if not words:
            print(f"{year}年没有有效关键词")
            continue

        # 统计词频
        counter = Counter(words)

        # 只保留前30个高频词，避免词云过于拥挤
        top_words = dict(counter.most_common(30))

        # 生成词云
        try:
            wc = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=30,
                colormap='viridis',
                relative_scaling=0.5,
                min_font_size=10
            ).generate_from_frequencies(top_words)

            plt.figure(figsize=(12, 6))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'{year}年论文关键词词云', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'wordclouds/wordcloud_{year}.png', dpi=300, bbox_inches='tight')
            plt.show()

            print(f"{year}年词云已保存为 wordclouds/wordcloud_{year}.png")
            print(f"{year}年高频词: {list(top_words.keys())[:10]}")  # 显示前10个高频词

        except Exception as e:
            print(f"{year}年词云生成失败: {e}")


def predict_and_visualize(df):
    """预测下一届论文数量并可视化"""
    if 'conference' not in df.columns or df.empty:
        print("没有数据，无法预测。")
        return

    predictions = {}
    actual_data = {}

    for i, conf in enumerate(df['conference'].unique()):
        sub = df[df['conference'] == conf]
        count = sub.groupby('year').size().reset_index()

        if len(count) < 2:
            print(f"{conf} 数据不足，无法预测")
            continue

        X = count['year'].values.reshape(-1, 1)
        y = count[0].values

        # 线性回归预测
        model = LinearRegression()
        model.fit(X, y)

        # 预测2025年
        next_year = 2025
        pred = model.predict([[next_year]])[0]
        predictions[conf] = int(pred)

        # 存储实际数据用于绘图
        actual_data[conf] = count

        print(f"{conf} 预测{next_year}年论文数：{int(pred)}")

    # 可视化预测结果
    if predictions:
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 左图：历史趋势和预测
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, (conf, data) in enumerate(actual_data.items()):
            # 绘制历史数据
            ax1.plot(data['year'], data[0], marker='o', label=f'{conf} (历史)',
                     linewidth=2, markersize=8, color=colors[i % len(colors)])

            # 绘制预测点
            if conf in predictions:
                ax1.scatter(2025, predictions[conf], marker='*', s=200,
                            color=colors[i % len(colors)], label=f'{conf} (预测)')

        ax1.set_xlabel('年份', fontsize=12)
        ax1.set_ylabel('论文数量', fontsize=12)
        ax1.set_title('历史趋势与2025年预测', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 右图：2025年预测柱状图
        confs = list(predictions.keys())
        pred_values = list(predictions.values())

        bars = ax2.bar(confs, pred_values, color=colors[:len(confs)], alpha=0.7)
        ax2.set_xlabel('会议名称', fontsize=12)
        ax2.set_ylabel('预测论文数量', fontsize=12)
        ax2.set_title('2025年各会议论文数量预测', fontsize=14, fontweight='bold')

        # 在柱状图上添加数值标签
        for bar, value in zip(bars, pred_values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(value), ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig('prediction.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("预测结果已保存为 prediction.png")


def main():
    """主函数"""
    print("=== 学术论文发表趋势分析 ===\n")

    # 1. 爬取数据
    df = get_all_papers()
    if df.empty:
        print("未获取到任何论文数据，请检查网络或爬虫逻辑。")
        return

    # 保存数据
    df.to_csv('papers.csv', index=False, encoding='utf-8-sig')
    print(f"\n数据已保存到 papers.csv")

    # 2. 绘制趋势图
    print("\n正在生成趋势图...")
    plot_trend(df)

    # 3. 生成年度词云
    print("\n正在生成年度词云图...")
    plot_yearly_wordclouds(df)

    # 4. 预测并可视化
    print("\n正在进行预测分析...")
    predict_and_visualize(df)

    print("\n=== 分析完成 ===")
    print("生成的文件：")
    print("- papers.csv: 论文数据")
    print("- trend.png: 趋势图")
    print("- wordclouds/: 年度词云图目录")
    print("- prediction.png: 预测结果图")


if __name__ == "__main__":
    main()