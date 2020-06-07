import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X_train = pd.read_csv("Linear_X_Train.csv")
y_train = pd.read_csv("Linear_Y_Train.csv")
X_test = pd.read_csv("Linear_X_Test.csv")

model = LinearRegression().fit(X_train, y_train)
y_test_predict = model.predict(X_test)

val = input("Enter the value to be tested: ")
test = np.array([val], dtype=np.float64).reshape(-1,1)
result = model.predict(test)
print(result)