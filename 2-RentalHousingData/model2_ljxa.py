import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV


def dummy_variables(df, columns):
    # 哑编码
    for col in columns:
        # 将 "unknown" 替换为 NaN
        df[col] = df[col].replace('unknown', np.nan)
        # 进行哑编码
        dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
        # 将原始列删除，并将编码列添加到 DataFrame 中
        df = df.drop(col, axis=1)
        df = pd.concat([df, dummies], axis=1)
    return df

def normalize_columns(df, columns):
    # 进行规范化
    scaler = MinMaxScaler()
    for col in columns:
        if df[col].dtype != 'object':
            # 从 DataFrame 中提取出该列，并将其转换为二维数组
            col_data = df[[col]].values
            # 调用 MinMaxScaler 对象的 fit_transform() 方法进行规范化
            normalized_data = scaler.fit_transform(col_data)
            # 将规范化后的数组转换回一维，并将其添加回 DataFrame 中
            df[col] = normalized_data.flatten()
    return df

def impute_missing_col_rf(df, col_name):
    # 使用随机森林预测缺失值
    # 分为已知和未知两部分
    known = df.loc[df[col_name].notnull()]
    unknown = df.loc[df[col_name].isnull()]

    # 构造训练集和测试集
    X_train = known.drop(col_name, axis=1)
    y_train = known[col_name]
    X_test = unknown.drop(col_name, axis=1)

    # 使用随机森林算法进行训练和预测
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    predicted = rf.predict(X_test)

    # 将预测结果填充到缺失值中
    df.loc[df[col_name].isnull(), col_name] = predicted

    return df


def split_and_model(df):
    # 分离自变量和因变量
    X = df.drop(columns=['rent_price'])
    y = df['rent_price']

    # 划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建线性回归模型对象
    model = LinearRegression()

    # 使用fit()方法进行拟合
    model.fit(X_train, y_train)

    # 对测试集进行预测
    y_pred = model.predict(X_test)

    # 计算MAE
    mae = mean_absolute_error(y_test, y_pred)
    print("MAE:",mae)


# 1 从csv文件中读取数据集
df = pd.read_csv('xablzufang_without_duplicates.csv',encoding='utf-8-sig')
df = df[['type', 'layout', 'bc', 'distance', 'rent_area', 'rent_price']]
print(df.head())
# # 统计每个字段各取值的个数
# for col in df.columns:
#     value_counts = df[col].value_counts()
#     print(f"列名: {col}")
#     print(value_counts)
#
#2 数据预处理
#2.1 对type的处理
df = df[df['type'] != '未知']
df['type']=df['type'].map({'整租':1,'合租':0})
pd.set_option('display.max_columns', 50)  # 显示完整的列
print(df['type'].value_counts())

#2.2 对layout的处理
df=df[df['layout']!='未知室0厅']
# 统计每个取值出现的频次
counts = df['layout'].value_counts()
# 获取出现频次低于50的取值列表
low_frequency = counts[counts < 50].index.tolist()
# 将出现频次低于50的取值合并为“其他”类别
df['layout'] = df['layout'].apply(lambda x: '其他' if x in low_frequency else x)

#2.3 对分类数据进行哑编码，对数值型数据进行规范化
categorical_columns = ['layout','bc']
numerical_columns=['rent_area','rent_price']

df=dummy_variables(df,categorical_columns)
df=normalize_columns(df,numerical_columns)

# 2.4 对distance的处理
#删除非None的数据中的单位“m”
df.loc[df['distance'] != 'None', 'distance'] = df.loc[df['distance'] != 'None', 'distance'].str[:-1].astype(float)

#对distance中“None”设置不同的处理方法
df1 = df.copy()
df2 = df.copy()
df3 = df.copy().replace('None', np.nan)
df4 = df.copy().replace('None', np.nan)
df1['distance']=df1["distance"].replace("None", "100000000").astype(float)#填充一个特别大的数
df2['distance']=df2["distance"].replace("None", "0").astype(float)#填充0
df3 = df3.fillna(df3.mean()) #填充平均数
# df4=impute_missing_col_rf(df4,'distance') #随机森林算法预测缺失值
# for df in [df1,df2,df3,df4]:
#     split_and_model(df)

#选择填充特别大的数据来处理缺失值
df['distance']=df["distance"].replace("None", "100000000").astype(float)
df=normalize_columns(df,['distance'])

# 3 确立最佳参数并建模
# 拆分特征和目标变量
X = df.drop('rent_price', axis=1)
y = df['rent_price']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 定义模型
model = LinearRegression()

# 定义参数范围
params = {'fit_intercept': [True, False],
          'copy_X': [True, False]}

# 定义评估指标
scoring = 'neg_mean_absolute_error'

# 定义交叉验证对象
grid = GridSearchCV(model, params, cv=5, scoring=scoring)

# 在训练集上进行交叉验证
grid.fit(X_train, y_train)

# 输出最优参数
print("最优参数：",grid.best_params_)

# 使用最优参数构建模型
best_model = LinearRegression(fit_intercept=grid.best_params_['fit_intercept'],
                               copy_X=grid.best_params_['copy_X'])

# 在测试集上评估模型表现
y_pred = best_model.fit(X_train, y_train).predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print('基于最优参数建立模型的MAE:', mae)