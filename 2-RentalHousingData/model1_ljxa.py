import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 从csv文件中读取数据集
data = pd.read_csv('xablzufang.csv', encoding='utf-8-sig')
col = ['type', 'bc', 'distance', 'rent_area', 'rent_price']
data = pd.DataFrame(data, columns=col)

# 将"None"替换为"100000000"，并将距离转换为数字类型
data["distance"] = data["distance"].replace("None", "100000000")
data["distance"] = data["distance"].str.replace("m", "").astype(float)

# 将rent_area列的值缩放到0到1的范围内
data["rent_area"] = data["rent_area"] / 5000.0

# 使用OneHotEncoder将类型和区域编码为二进制形式
encoder = OneHotEncoder(sparse=False)
data_encoded = pd.DataFrame(encoder.fit_transform(data[["type", "bc"]]))
data_encoded.columns = encoder.get_feature_names(["type", "bc"])
data = pd.concat([data, data_encoded], axis=1)

# 删除原始的"type"和"bc"列
data = data.drop(["type", "bc"], axis=1)

# 将数据集拆分为训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2)

# 训练线性回归模型
model = LinearRegression()
model.fit(train_data.drop("rent_price", axis=1), train_data["rent_price"])

# 对测试集进行预测，并计算MAE
predictions = model.predict(test_data.drop("rent_price", axis=1))
mae = mean_absolute_error(test_data["rent_price"], predictions)
print("MAE:", mae)
