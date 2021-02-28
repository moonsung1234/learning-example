
# run with Decision tree algorithm

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV # RandomizedSearchCV is used instead of this.
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from scipy.stats import uniform, randint
import pandas as pd
import numpy as np

wine_data = pd.read_csv("https://bit.ly/wine-date")
x_all = wine_data[["alcohol", "sugar", "pH"]].to_numpy()
t_all = wine_data["class"].to_numpy()

# prac data : 60%
# ver data : 20%
# test data : 20%

train_x, test_x, train_t, test_t = train_test_split(x_all, t_all, test_size=0.2, random_state=2021)
prac_x, ver_x, prac_t, ver_t = train_test_split(train_x, train_t, test_size=0.2, random_state=2021)

params = {
    "min_impurity_decrease" : uniform(0.0001, 0.001),
    "max_depth" : randint(5, 20),
    "min_samples_split" : randint(5, 100),
    "min_samples_leaf" : randint(5, 100)
}

dt = DecisionTreeClassifier(random_state=2021)
gs = RandomizedSearchCV(dt, params, n_iter=100, n_jobs=-1, random_state=2021)
gs.fit(train_x, train_t)

# best 파라미터 & 점수 확인하기
print(gs.best_params_)
print(gs.best_score_)

# 교차검증 예제(그리드서치에서 다 이루어짐.)
# splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=2021)
# scores = cross_validate(dt, test_x, text_t, cv=splitter)
# print(np.mean(scores["test_score"]))