
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from scipy.special import expit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fish = pd.read_csv("https://bit.ly/fish_csv")
x_all = fish[["Weight", "Length", "Diagonal", "Height", "Width"]].to_numpy()
t_all = fish["Species"].to_numpy()

train_x, test_x, train_t, test_t = train_test_split(x_all, t_all, random_state=2021)

ss = StandardScaler()
ss.fit(train_x)

train_scaled_x = ss.transform(train_x)
test_scaled_x = ss.transform(test_x)

lr = LogisticRegression()
lr.fit(train_scaled_x, train_t)

# score 출력
print(lr.score(train_scaled_x, train_t))
print(lr.score(test_scaled_x, test_t))

# 앞 5개 예측값 출력
print("predict : ", lr.predict(train_scaled_x[:5]))
print("target : ", train_t[:5])

print("predict : ", lr.predict(test_scaled_x[:5]))
print("target : ", test_t[:5])

# 확률 출력
print("train : \n", np.round(lr.predict_proba(train_scaled_x[:5]), decimals=3))
print("test : \n", np.round(lr.predict_proba(test_scaled_x[:5]), decimals=3))
print("class : ", lr.classes_)
