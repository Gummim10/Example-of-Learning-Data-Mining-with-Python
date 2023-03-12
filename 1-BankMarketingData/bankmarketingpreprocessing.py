#data preprocessing

import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


def transform_to_binary(df, col_name):
    # yes->1   no -> 0   unknown -> NaN
    df[col_name] = df[col_name].map({'yes': 1, 'no': 0})  # 将"yes"值替换为1，将"no"值替换为0
    df[col_name].replace('unknown', np.nan, inplace=True)  # 将"unknown"替换为NaN
    return df

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


def impute_missing_values(df, column_list):
    # 创建训练集和测试集
    train_data = df[df[column_list].notnull().all(axis=1)]
    test_data = df[df[column_list].isnull().any(axis=1)]

    # 分离特征和标签
    X_train = train_data.drop(column_list, axis=1)
    y_train = train_data[column_list]

    X_test = test_data.drop(column_list, axis=1)

    # 使用随机森林模型拟合
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    # 预测缺失值
    y_pred = rf.predict(X_test)

    # 填充缺失值
    df.loc[df[column_list].isnull().any(axis=1), column_list] = y_pred

    # 返回填充后的数据
    return df

# 1 导入数据
df = pd.read_csv('bank-additional-full.csv', encoding='utf-8-sig', sep=';')
pd.set_option('display.max_columns', 100)  # 显示完整的列
# print(df.head(30))
# print(df.shape)


# 2 数据预处理与特征工程
# 2.1 分类数据数值化（哑编码）
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                       'poutcome', 'y']  # 分类数据
numerical_columns = list(set(df.columns) - set(categorical_columns))  # 数值数据

# 二元分类数据
categoricaltobinarycols = ['default', 'housing', 'loan', 'y']
for col in categoricaltobinarycols:
    df = transform_to_binary(df, col)
# 分类数据哑编码
df = dummy_variables(df, ['job', 'marital', 'education', 'contact', 'month', 'day_of_week', 'poutcome'])
# 2.2 数值型数据标准化
df = normalize_columns(df, numerical_columns)

# 2.3 缺失值处理

# # 查看数值型属性缺失值
# for col in numerical_columns:
#     print(col + " : " + str(df[col].isnull().sum()))
#
# # 查看字符型属性缺失值
# df.replace('unknown', np.nan, inplace=True)  #将"unknown"替换为NaN
# for col in categorical_columns:
#     print(col + " : " + str(df[col].isnull().sum()))


# 删除job和marital字段中的缺失值

job_cols = [col for col in df.columns if re.search('^job_', col)] # 使用正则表达式查找所有以 "job" 开头的列名
for col in job_cols:
    df.dropna(subset=[col], inplace=True)

marital_cols =[col for col in df.columns if re.search('^marital_', col)] # 使用正则表达式查找所有以 "marital" 开头的列名
for col in job_cols:
    df.dropna(subset=[col], inplace=True)

# 使用随机森林算法来预测缺失值
col_list = ['housing', 'loan', 'default']
education_cols = [col for col in df.columns if re.search('^education_', col)]
col_list += education_cols
df = impute_missing_values(df, col_list)

# 将经过处理后的 df 保存为 CSV 文件
df.to_csv('processed_data.csv', index=True)