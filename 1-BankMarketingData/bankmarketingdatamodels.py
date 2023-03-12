# models

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

# 1.导入数据
df = pd.read_csv('processed_data.csv', encoding='utf-8-sig')
pd.set_option('display.max_columns', 100)
df = df.drop(df.columns[0], axis=1)  # 原始数据中第一列为列号，导入后需要去掉
# print(df.head(30))

# 2.划分数据集
train_data, test_val_data = train_test_split(df, test_size=0.4,
                                             random_state=42)  # 使用train_test_split函数将原始数据集data按照test_size=0.4的比例划分为训练集train_data和测试集+验证集的组合test_val_data，其中test_size表示测试集所占比例，这里设置为40%。
test_data, val_data = train_test_split(test_val_data, test_size=0.5,
                                       random_state=42)  # 使用train_test_split函数将test_val_data按照test_size=0.5的比例划分为测试集test_data和验证集val_data
# 打印数据集大小
print(f'Train data size: {train_data.shape[0]}')
print(f'Validation data size: {val_data.shape[0]}')
print(f'Test data size: {test_data.shape[0]}')

# 简单统计查看数据是否平衡
print(df['y'].value_counts())

# 对训练集数据进行SMOTE

X_train = train_data.drop("y", axis=1)
y_train = train_data["y"]

smote = SMOTE(random_state=42, k_neighbors=3)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print('原始数据集中各类别样本数量：\n', y_train.value_counts())
print('过采样后数据集中各类别样本数量：\n', y_train_res.value_counts())

X_train_res = X_train_res.values
y_train_res = y_train_res.values

X_test = test_data.drop("y", axis=1).values
y_test = test_data["y"].values

X_val = val_data.drop("y", axis=1).values
y_val = val_data["y"].values

# 3.0 定义评价指标为正样本的F1-score
scoring = {'f1_score': make_scorer(f1_score, pos_label=1)}

# #3.1在训练集上建立逻辑回归模型(Logistic Regression)，并寻找最优参数
#
# # 创建Logistic Regression对象
# lr_model = LogisticRegression(max_iter = 100000)
#
# # 定义要搜索的参数组合
# parameters_lr = {
#     'solver': ['lbfgs', 'liblinear'],
#     'penalty': ['l1', 'l2'],
#     'C': [0.1, 1, 10]
# }

# # 创建GridSearchCV对象
# grid_search_lr = GridSearchCV(lr_model, parameters_lr, scoring=scoring, cv=5, return_train_score=True, refit='f1_score')
#
# # 在训练集上拟合GridSearchCV对象
# grid_search_lr.fit(X_train_res, y_train_res)
#
# # 输出最优参数及其对应的分数
# print("Model name: Logistic Regression")
# print("Best parameters: ", grid_search_lr.best_params_)
# print('Best Score:', grid_search_lr.best_score_)

# #3.2在训练集上建立朴素贝叶斯模型(Naive Bayes)，并寻找最优参数
#
# # 设置参数范围
# parameters_clf = {
#     'var_smoothing': np.logspace(0,-9, num=100)
# }
#
# # 创建一个GaussianNB分类器对象
# clf = GaussianNB()
#
# # 创建GridSearchCV对象
# grid_search_clf = GridSearchCV(clf, parameters_clf, scoring=scoring, cv=5,return_train_score=True, refit='f1_score')
#
# # 在训练集上拟合GridSearchCV对象
# grid_search_clf.fit(X_train_res,y_train_res)
#
# # 打印最优参数和最优得分
# print("Model name: Naive Bayes")
# print("Best parameters: {}".format(grid_search_clf.best_params_))
# print("Best F1-score: {:.2f}".format(grid_search_clf.best_score_))

# #3.3在训练集上建立随机森林模型(Random Forest)，并寻找最优参数
#
# # 定义参数范围
# parameters_rfc = {'n_estimators': [50, 100, 200],
#           'max_depth': [5, 10, 20],
#           'min_samples_split': [2, 5, 10],
#           'min_samples_leaf': [1, 2, 4]}
#
# # 创建模型
# rfc = RandomForestClassifier()
#
# # 创建GridSearchCV对象
# grid_search_rfc = GridSearchCV(estimator=rfc, param_grid=parameters_rfc, scoring=scoring, cv=5,return_train_score=True, refit='f1_score')
#
# # 在训练集上拟合GridSearchCV对象
# grid_search_rfc.fit(X_train_res,y_train_res)
#
# # 输出最优参数和最优得分
# print("Model name: Random Forest")
# print('Best parameters:', grid_search_rfc.best_params_)
# print('Best score:', grid_search_rfc.best_score_)

# # 3.4 在训练集上建立支持向量机(SVM)，并寻找最优参数
# # 定义参数范围
# parameters_svc = {
#     'C': np.logspace(-2, 2, 5),
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     'gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 7))
# }
# # 定义模型
# svc = svm.SVC()
#
# # 定义GridSearchCV
# grid_search_svc = GridSearchCV(svc, param_grid=parameters_svc, scoring=scoring, cv=5, return_train_score=True,
#                                refit='f1_score', n_jobs=-1)
#
# # 训练模型
# grid_search_svc.fit(X_train_res, y_train_res)
#
# # 输出最佳参数和最佳得分
# print("Model name: Support Vector Machine")
# print("Best parameters: ", grid_search_svc.best_params_)
# print("Best score: ", grid_search_svc.best_score_)

#4 在交叉验证集上测试各模型的性能
best_lr_model = LogisticRegression(C=0.1, penalty='l1', solver='liblinear', max_iter=100000)
best_clf_model = GaussianNB(var_smoothing=1.519911082952933e-06)
best_rfc_model= RandomForestClassifier(max_depth =20,min_samples_leaf=1,min_samples_split=2,n_estimators = 100)

models = [best_lr_model, best_clf_model, best_rfc_model]

best_lr_model.fit(X_train_res,y_train_res)
best_clf_model.fit(X_train_res,y_train_res)
best_rfc_model.fit(X_train_res,y_train_res)

for model in models:
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, pos_label=1)
    print("模型", model.__class__.__name__, "在val上正样本的F1分数为：", f1)

#5 在测试集上应用随机森林 RandomForestClassifier
y_pred_test = best_rfc_model.predict(X_test)
f1 = f1_score(y_test, y_pred_test, pos_label=1)
print("模型", model.__class__.__name__, "在测试集上正样本的F1分数为：", f1)