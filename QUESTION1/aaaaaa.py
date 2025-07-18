import matplotlib.pyplot as plt
import requests
import pandas as pd
from time import sleep
import random
import numpy as np
import seaborn as sns
from collections import Counter

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 初始化数据列表
name = []
gender = []
age = []
birthplace = []
wealth = []
ranking = []
com_name = []
ind_name = []

# 爬取数据
for page in range(1, 11):
    s_sec = random.uniform(1, 2)
    print('等待{}秒'.format(s_sec))
    sleep(s_sec)
    print('爬取{}页'.format(page))
    offset = (page - 1) * 100
    url = 'https://www.hurun.net/zh-CN/Rank/HsRankDetailsList?num=ODBYW2BI&search=&offset={}&limit=100'.format(offset)

    # 构造请求头
    header = {
        'User-Agent': 'Mozilla/5.0 (Linux;Android 6.0;Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36',
        'Accept': 'application/json,text/javascript,*/*;q=0.01',
        'Accept-Language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        'Accept-Encoding': 'gzip, deflate,br',
        'Content-Type': 'application/json',
        'referer': 'https://www.hurun.net/zh-CN/Rank/HsRankDetails?pagetype=rich'
    }

    # 发送请求
    res = requests.get(url, headers=header)
    j_data = res.json()
    rows = j_data.get('rows', [])

    # 解析数据
    for row in rows:
        r = row["hs_Character"][0]
        name.append(row.get("hs_Rank_Rich_ChaName_Cn"))
        gender.append(r.get("hs_Character_Gender"))
        age.append(r.get("hs_Character_Age"))
        birthplace.append(r.get("hs_Character_BirthPlace_Cn"))
        wealth.append(row.get("hs_Rank_Rich_Wealth"))
        ranking.append(row.get("hs_Rank_Rich_Ranking"))
        com_name.append(row.get("hs_Rank_Rich_ComName_Cn"))
        ind_name.append(row.get("hs_Rank_Rich_Industry_Cn"))

# 拼装数据
df = pd.DataFrame({
    '排名': ranking,
    '姓名': name,
    '性别': gender,
    '年龄': age,
    '出生地': birthplace,
    '财富': wealth,
    '公司名称': com_name,
    '行业名称': ind_name
})

# 保存数据
df.to_csv('胡润富豪榜.csv', index=False, encoding='utf-8-sig')
print(f"数据已保存到 胡润富豪榜.csv，共{len(df)}条记录")


# 数据预处理
df['财富'] = pd.to_numeric(df['财富'], errors='coerce')
df['年龄'] = pd.to_numeric(df['年龄'], errors='coerce')

# 1. 各行业富豪数量统计
industry_count = df['行业名称'].value_counts()
print("\n各行业富豪数量统计（前10名）：")
print(industry_count.head(10))

# 2. 各行业财富总值统计
industry_wealth = df.groupby('行业名称')['财富'].sum().sort_values(ascending=False)
print("\n各行业财富总值统计（前10名）：")
print(industry_wealth.head(10))

# 3. 各行业平均财富统计
industry_avg_wealth = df.groupby('行业名称')['财富'].mean().sort_values(ascending=False)
print("\n各行业平均财富统计（前10名）：")
print(industry_avg_wealth.head(10))

# 可视化：行业富豪数量分布
plt.figure(figsize=(15, 10))

# 子图1：行业富豪数量柱状图
plt.subplot(2, 2, 1)
top_industries = industry_count.head(10)
bars = plt.bar(range(len(top_industries)), top_industries.values, color='skyblue')
plt.title('各行业富豪数量分布（前10名）', fontsize=14, fontweight='bold')
plt.xlabel('行业名称')
plt.ylabel('富豪数量')
plt.xticks(range(len(top_industries)), top_industries.index, rotation=45, ha='right')
for i, bar in enumerate(bars):
    height = bar.get_height()
    if not pd.isna(height):  # 检查是否为NaN
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                 str(int(height)), ha='center', va='bottom')

# 子图2：行业财富总值柱状图
plt.subplot(2, 2, 2)
top_wealth_industries = industry_wealth.head(10)
bars = plt.bar(range(len(top_wealth_industries)), top_wealth_industries.values, color='lightgreen')
plt.title('各行业财富总值分布（前10名）', fontsize=14, fontweight='bold')
plt.xlabel('行业名称')
plt.ylabel('财富总值（亿元）')
plt.xticks(range(len(top_wealth_industries)), top_wealth_industries.index, rotation=45, ha='right')
for i, bar in enumerate(bars):
    height = bar.get_height()
    if not pd.isna(height):  # 检查是否为NaN
        plt.text(bar.get_x() + bar.get_width() / 2, height + 50,
                 f'{int(height):,}', ha='center', va='bottom')

# 子图3：行业平均财富柱状图
plt.subplot(2, 2, 3)
top_avg_wealth = industry_avg_wealth.head(10)
bars = plt.bar(range(len(top_avg_wealth)), top_avg_wealth.values, color='orange')
plt.title('各行业平均财富分布（前10名）', fontsize=14, fontweight='bold')
plt.xlabel('行业名称')
plt.ylabel('平均财富（亿元）')
plt.xticks(range(len(top_avg_wealth)), top_avg_wealth.index, rotation=45, ha='right')
for i, bar in enumerate(bars):
    height = bar.get_height()
    if not pd.isna(height):  # 检查是否为NaN
        plt.text(bar.get_x() + bar.get_width() / 2, height + 5,
                 f'{int(height):,}', ha='center', va='bottom')

# 子图4：行业富豪数量vs平均财富散点图
plt.subplot(2, 2, 4)
industry_stats = pd.DataFrame({
    '富豪数量': industry_count,
    '平均财富': industry_avg_wealth,
    '财富总值': industry_wealth
}).dropna()

plt.scatter(industry_stats['富豪数量'], industry_stats['平均财富'],
            s=industry_stats['财富总值'] / 100, alpha=0.6, c='red')
plt.title('行业富豪数量 vs 平均财富关系', fontsize=14, fontweight='bold')
plt.xlabel('富豪数量')
plt.ylabel('平均财富（亿元）')

plt.tight_layout()
plt.savefig('行业分析.png', dpi=300, bbox_inches='tight')
plt.show()

# 数据清洗
df_clean = df.dropna(subset=['年龄', '性别', '出生地', '财富']).copy()

# 1. 性别分布分析
gender_dist = df_clean['性别'].value_counts()
print("\n性别分布：")
print(gender_dist)

# 2. 年龄分布分析
age_bins = [0, 30, 40, 50, 60, 70, 100]
age_labels = ['30岁以下', '30-40岁', '40-50岁', '50-60岁', '60-70岁', '70岁以上']
df_clean.loc[:, '年龄分组'] = pd.cut(df_clean['年龄'], bins=age_bins, labels=age_labels, right=False)
age_dist = df_clean['年龄分组'].value_counts()
print("\n年龄分布：")
print(age_dist)

# 3. 出生地分布分析
birthplace_dist = df_clean['出生地'].value_counts().head(10)
print("\n出生地分布（前10名）：")
print(birthplace_dist)

# 4. 财富分布分析
wealth_bins = [0, 100, 200, 500, 1000, 5000, float('inf')]
wealth_labels = ['100亿以下', '100-200亿', '200-500亿', '500-1000亿', '1000-5000亿', '5000亿以上']
df_clean.loc[:, '财富分组'] = pd.cut(df_clean['财富'], bins=wealth_bins, labels=wealth_labels, right=False)
wealth_dist = df_clean['财富分组'].value_counts()
print("\n财富分布：")
print(wealth_dist)

# 可视化：多维度分析
plt.figure(figsize=(20, 15))

# 子图1：性别分布饼图
plt.subplot(3, 3, 1)
plt.pie(gender_dist.values, labels=gender_dist.index, autopct='%1.1f%%', startangle=90)
plt.title('富豪性别分布', fontsize=14, fontweight='bold')

# 子图2：年龄分布柱状图
plt.subplot(3, 3, 2)
bars = plt.bar(range(len(age_dist)), age_dist.values, color='lightcoral')
plt.title('富豪年龄分布', fontsize=14, fontweight='bold')
plt.xlabel('年龄分组')
plt.ylabel('人数')
plt.xticks(range(len(age_dist)), age_dist.index, rotation=45)
for i, bar in enumerate(bars):
    height = bar.get_height()
    if not pd.isna(height):  # 检查是否为NaN
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1,
                 str(int(height)), ha='center', va='bottom')

# 子图3：出生地分布柱状图
plt.subplot(3, 3, 3)
top_birthplaces = birthplace_dist.head(8)
bars = plt.bar(range(len(top_birthplaces)), top_birthplaces.values, color='lightblue')
plt.title('富豪出生地分布（前8名）', fontsize=14, fontweight='bold')
plt.xlabel('出生地')
plt.ylabel('人数')
plt.xticks(range(len(top_birthplaces)), top_birthplaces.index, rotation=45, ha='right')
for i, bar in enumerate(bars):
    height = bar.get_height()
    if not pd.isna(height):  # 检查是否为NaN
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                 str(int(height)), ha='center', va='bottom')

# 子图4：财富分布柱状图
plt.subplot(3, 3, 4)
bars = plt.bar(range(len(wealth_dist)), wealth_dist.values, color='lightgreen')
plt.title('富豪财富分布', fontsize=14, fontweight='bold')
plt.xlabel('财富分组')
plt.ylabel('人数')
plt.xticks(range(len(wealth_dist)), wealth_dist.index, rotation=45)
for i, bar in enumerate(bars):
    height = bar.get_height()
    if not pd.isna(height):  # 检查是否为NaN
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                 str(int(height)), ha='center', va='bottom')

# 子图5：性别vs平均财富
plt.subplot(3, 3, 5)
gender_wealth = df_clean.groupby('性别')['财富'].mean()
bars = plt.bar(gender_wealth.index, gender_wealth.values, color=['pink', 'lightblue'])
plt.title('不同性别平均财富对比', fontsize=14, fontweight='bold')
plt.ylabel('平均财富（亿元）')
for i, bar in enumerate(bars):
    height = bar.get_height()
    if not pd.isna(height):  # 检查是否为NaN
        plt.text(bar.get_x() + bar.get_width() / 2, height + 10,
                 f'{int(height):,}', ha='center', va='bottom')

# 子图6：年龄vs平均财富
plt.subplot(3, 3, 6)
age_wealth = df_clean.groupby('年龄分组', observed=False)['财富'].mean()
bars = plt.bar(range(len(age_wealth)), age_wealth.values, color='gold')
plt.title('不同年龄组平均财富对比', fontsize=14, fontweight='bold')
plt.xlabel('年龄分组')
plt.ylabel('平均财富（亿元）')
plt.xticks(range(len(age_wealth)), age_wealth.index, rotation=45)
for i, bar in enumerate(bars):
    height = bar.get_height()
    if not pd.isna(height):  # 检查是否为NaN
        plt.text(bar.get_x() + bar.get_width() / 2, height + 10,
                 f'{int(height):,}', ha='center', va='bottom')

# 子图7：年龄分布热力图
plt.subplot(3, 3, 7)
age_gender_cross = pd.crosstab(df_clean['年龄分组'], df_clean['性别'])
sns.heatmap(age_gender_cross, annot=True, fmt='d', cmap='YlOrRd')
plt.title('年龄-性别分布热力图', fontsize=14, fontweight='bold')

# 子图8：财富-年龄散点图
plt.subplot(3, 3, 8)
plt.scatter(df_clean['年龄'], df_clean['财富'], alpha=0.6, c='purple')
plt.title('财富与年龄关系散点图', fontsize=14, fontweight='bold')
plt.xlabel('年龄')
plt.ylabel('财富（亿元）')

# 子图9：行业-性别分布
plt.subplot(3, 3, 9)
top_industries_gender = df_clean[df_clean['行业名称'].isin(industry_count.head(5).index)]
industry_gender_cross = pd.crosstab(top_industries_gender['行业名称'], top_industries_gender['性别'])
industry_gender_cross.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('主要行业性别分布', fontsize=14, fontweight='bold')
plt.xlabel('行业名称')
plt.ylabel('人数')
plt.xticks(rotation=45, ha='right')
plt.legend(title='性别')

plt.tight_layout()
plt.savefig('多维度分析.png', dpi=300, bbox_inches='tight')
plt.show()



print(f"\n1. 数据概况：")
print(f"   - 总富豪数量：{len(df)}人")
print(f"   - 有效数据：{len(df_clean)}人")
print(f"   - 涉及行业：{len(industry_count)}个")

print(f"\n2. 行业分析：")
print(f"   - 富豪最多的行业：{industry_count.index[0]}（{industry_count.iloc[0]}人）")
print(f"   - 财富总值最高的行业：{industry_wealth.index[0]}（{int(industry_wealth.iloc[0]):,}亿元）")
print(f"   - 平均财富最高的行业：{industry_avg_wealth.index[0]}（{int(industry_avg_wealth.iloc[0]):,}亿元）")

print(f"\n3. 人口统计：")
print(f"   - 男性富豪：{gender_dist.get('先生', 0)}人（{gender_dist.get('先生', 0) / len(df_clean) * 100:.1f}%）")
print(f"   - 女性富豪：{gender_dist.get('女士', 0)}人（{gender_dist.get('女士', 0) / len(df_clean) * 100:.1f}%）")
print(f"   - 平均年龄：{df_clean['年龄'].mean():.1f}岁")
print(f"   - 最年轻富豪：{df_clean['年龄'].min():.0f}岁")
print(f"   - 最年长富豪：{df_clean['年龄'].max():.0f}岁")

print(f"\n4. 财富统计：")
print(f"   - 平均财富：{df_clean['财富'].mean():.0f}亿元")
print(f"   - 最高财富：{df_clean['财富'].max():.0f}亿元")
print(f"   - 最低财富：{df_clean['财富'].min():.0f}亿元")

print(f"\n5. 地域分布：")
print(f"   - 富豪最多的省份：{birthplace_dist.index[0]}（{birthplace_dist.iloc[0]}人）")

print("\n生成的文件：")
print("- 胡润富豪榜.csv：原始数据")
print("- 行业分析.png：行业分析图表")
print("- 多维度分析.png：多维度分析图表")