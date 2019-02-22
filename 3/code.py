"""多元线性回归"""
# Step 1: 数据预处理
# 导入库
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy
import pandas as pd
import os

# 导入数据集
path = os.path.join(os.path.dirname(__file__), '50_Startups.csv')
dataset = pd.read_csv(path)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# 将类别数据数字化
labelencoder = LabelEncoder()
X[:, -1] = labelencoder.fit_transform(X[:, -1])
onehotencoder = OneHotEncoder(categorical_features=[-1])
X = onehotencoder.fit_transform(X).toarray()

# 避免虚拟变量陷阱
X = X[:, 1:]

# 拆分数据集为训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Step 2: 在训练集上训练多元线性回归模型
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Step 3: 在测试集上预测结果
Y_pred = regressor.predict(X_test)
