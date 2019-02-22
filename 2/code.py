# 1: 数据预处理
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
"""简单线性回归"""
import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = os.path.join(os.path.dirname(__file__), 'studentscores.csv')
dataset = pd.read_csv(file_path)
x = dataset.iloc[:, :1].values
y = dataset.iloc[:, 1].values
x_train, x_test, y_train, y_test = train_test_split(x, y)

# 2: 训练集使用简单线性回归模型训练
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# 3: 预测结果
y_pred = regressor.predict(x_test)

# 4: 可视化
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.show()

plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, regressor.predict(x_test), color='blue')
plt.show()
