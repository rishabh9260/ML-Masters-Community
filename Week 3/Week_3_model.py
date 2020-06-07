import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Car_Purchasing_Data.csv", encoding='latin-1')
X = df.iloc[:, 3:-1]
y = pd.DataFrame(df.iloc[:, -1])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

## Correlation Plotting
# a = df[X.columns.tolist() + y.columns.tolist()]
# corrmat = a.corr()
# f, ax = plt.subplots(figsize =(9, 9)) 
# sns.heatmap(corrmat, ax = ax, cmap ="rocket_r", linewidths = 0.3, annot = True, fmt = ".2f") 
# plt.show()

# ## Standardizing the data  # Invariant to Normalizing
# scaler = StandardScaler()
# scaler.fit_transform(X_train)
# scaler.transform(X_test)

## The Model
model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))