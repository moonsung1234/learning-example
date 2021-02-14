
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

csv = pd.read_csv("https://bit.ly/perch_csv")
perch_lhw_data = csv.to_numpy()

perch_weight = np.array([
    5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
    115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
    150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
    218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
    556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
    850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
    1000.0])

train_x, test_x, train_t, test_t = train_test_split(perch_lhw_data, perch_weight, random_state=2021)

poly = PolynomialFeatures(include_bias=False)
poly.fit(train_x)
train_poly_x = poly.transform(train_x)
test_poly_x = poly.transform(test_x)

ss = StandardScaler()
ss.fit(train_poly_x)
train_scaled_x = ss.transform(train_poly_x)
test_scaled_x = ss.transform(test_poly_x)

# print(test_scaled_x)

# 릿지
ridge = Ridge()
ridge.fit(train_scaled_x, train_t)

# 릿지 스코어
print("train score : ", ridge.score(train_scaled_x, train_t))
print("test score : ", ridge.score(test_scaled_x, test_t))

# 라소
lasso = Lasso(max_iter=10000)
lasso.fit(train_scaled_x, train_t)

# 라소 스코어
print("train score : ", lasso.score(train_scaled_x, train_t))
print("test score : ", lasso.score(test_scaled_x, test_t))