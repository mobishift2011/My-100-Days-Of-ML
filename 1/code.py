"""数据预处理"""
# Step 1: Importing libraries
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import numpy as np
import pandas as pd
import os

# Step 2: Importing dataset
file_path = os.path.join(os.path.relpath(
    os.path.dirname(__file__)), 'Data.csv')
dataset = pd.read_csv(file_path)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
print("Step 2: Importing dataset")
print("X")
print(X)
print("Y")
print(Y)

# Step 3: Handling the missing data
imputer = SimpleImputer()
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
print("---------------------")
print("Step 3: Handling the missing data")
print("step2")
print("X")
print(X)


# Step 4: Encoding categorical data
label_encoder = LabelEncoder()
X[:, 0] = label_encoder.fit_transform(X[:, 0])
X = OneHotEncoder(categories='auto').fit_transform(X).toarray()
Y = label_encoder.fit_transform(Y)
print("---------------------")
print("Step 4: Encoding categorical data")
print("X")
print(X)
print("Y")
print(Y)

# Step 5: Splitting the datasets into training sets and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print("---------------------")
print("Step 5: Splitting the datasets into training sets and Test sets")
print("X_train")
print(x_train)
print("X_test")
print(x_test)
print("Y_train")
print(y_train)
print("Y_test")
print(y_test)

# Step 6. Feature scaling
sc = StandardScaler()
sc.fit_transform(x_train)
sc.transform(x_test)
print("---------------------")
print("Step 6: Feature Scaling")
print("X_train")
print(x_train)
print("X_test")
print(x_test)
