import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

# 0 去掉重复值
# 读取CSV文件
df = pd.read_csv('xablzufang.csv',encoding='utf-8-sig')

# 判断是否有重复的网址链接，如果有则删除
if df['url'].duplicated().any():
    df.drop_duplicates(subset=['url'], inplace=True)

# # 将结果保存为新的CSV文件
# df.to_csv('xablzufang_without_duplicates.csv', index=False,encoding='utf-8-sig')

# 1 租房房源分布
    # a:地图热力图（高德）
    # b:二级区域（商圈）租房房源条形图

# # 统计不同地区的房源数量
# counts = df['bc'].value_counts()
#
# # 绘制条形图
# plt.bar(counts.index, counts.values)
# plt.title('碑林区租房房源分布')
# plt.xlabel('地区')
# plt.ylabel('房源数量')
# plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei'] # 设置字体
# plt.savefig('房源数量条形图.png')  # 保存图像
# plt.show()  # 显示图像


#2租房租金分布-条形图

# 按地区分组，并计算平均租金
grouped = df.groupby('bc').mean()['aver_price']

# 按平均租金进行排序
grouped = grouped.sort_values()

# 绘制条形图
fig, ax = plt.subplots(figsize=(8,6))
ax.barh(grouped.index, grouped.values)
ax.set_xlabel('平均租金（元/平米·月）')
ax.set_ylabel('地区')
ax.set_title('不同地区平均租金')
plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei'] # 设置字体
plt.tight_layout()
plt.savefig('aver_price_by_bc.png')
plt.show()

#2 租房租金分布-箱线图 （aver_price:每平方米租金）

# 按地区分组，计算每个地区的平均租金
grouped = df.groupby('bc')['aver_price'].mean()

# 将Series转换成DataFrame
grouped_df = grouped.to_frame().reset_index()

# 绘制箱线图
plt.figure(figsize=(8, 6))
sns.boxplot(x='bc', y='aver_price', data=df, order=grouped_df['bc'])
plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']
plt.xlabel('地区')
plt.ylabel('平均租金')
plt.title('不同地区平均租金的箱线图')
plt.savefig('aver_price_boxplot.png')
plt.show()



# #3 距离最近地铁站距离和租金的关系
#
# # 去掉距离字段中的'm'
# df['distance'] = df['distance'].str.strip('m')
#
# # 删除值为'None'的整行数据
# df_clean = df[df['distance'] != 'None']
#
# # 将distance字段转换为float类型
# df_clean['distance'] = df_clean['distance'].astype(float)
#
# # 将aver_price字段转换为float类型
# df_clean['aver_price'] = df_clean['aver_price'].astype(float)
#
# # 计算距离最近地铁站距离和每平方米租金的相关系数
# corr = np.corrcoef(df_clean['distance'], df_clean['aver_price'])[0][1]
#
# # 输出相关系数和p-value
# print('距离最近地铁站距离和每平方米租金的相关系数为: {:.2f}'.format(corr))

# # 绘制散点图
# plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei'] # 设置字体
# plt.scatter(df_clean['distance'], df_clean['aver_price'])
# plt.xlabel('距离最近地铁站距离')
# plt.ylabel('每平方米租金')
# plt.title('距离最近地铁站距离与每平方米租金的关系')
# plt.savefig('distance_price_scatter.png')
# plt.show()

#4 整租：房屋大小与每平米租金的关系

# 数据清洗
df_clean = df[df['type'] == '合租']  # 选取type为整租的数据
df_clean2 = df_clean[(df_clean['rent_area'] <= 60) & (df_clean['rent_area'] >= 0)]  # 删除rent_area超过200和小于0的数据

# 绘制散点图
plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei'] # 设置字体
plt.scatter(df_clean2['rent_area'], df_clean2['aver_price'], alpha=0.5)
plt.xlabel('租赁面积', fontsize=14)
plt.ylabel('每平米租金', fontsize=14)
plt.xticks(range(0, 61, 5))
plt.title('合租租赁面积与每平米租金散点图', fontsize=16)

# 保存图片
plt.savefig('合租面积-租金scatter_plot.png')

# 显示图形
plt.show()




