
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

wine_data = pd.read_csv("https://bit.ly/wine-date")
x_all = wine_data[["alcohol", "sugar", "pH"]].to_numpy()
t_all = wine_data["class"].to_numpy()

train_x, test_x, train_t, test_t = train_test_split(x_all, t_all, test_size=0.25)

# 전처리할 필요 x
# ss = StandardScaler()
# ss.fit(train_x)
# train_scaled = ss.transform(train_x)
# test_scaled = ss.transform(test_x)

dt = DecisionTreeClassifier(max_depth=3, random_state=2021) 
dt.fit(train_x, train_t)

# score 출력
print("train data : ", dt.score(train_x, train_t))
print("test data : ", dt.score(test_x, test_t))

# tree 출력
plt.figure(figsize=(10, 7))
plot_tree(dt, filled=True, feature_names=["alcohol", "sugar", "pH"])
plt.show()