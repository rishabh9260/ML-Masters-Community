import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")
X = df.iloc[:, :-1]
y = df['target']

model = LinearRegression().fit(X, y)
y_predict = model.predict(test)