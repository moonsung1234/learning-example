
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

loop_count = 500
loop_range = 50

fish = pd.read_csv("https://bit.ly/fish_csv")
x_all = fish[["Weight", "Length", "Diagonal", "Height", "Width"]].to_numpy()
t_all = fish["Species"].to_numpy()

train_x, test_x, train_t, test_t = train_test_split(x_all, t_all, random_state=2021)

ss = StandardScaler()
ss.fit(train_x)

train_scaled = ss.transform(train_x)
test_scaled = ss.transform(test_x)

sc = SGDClassifier(loss="log", max_iter=100, random_state=2021)
sc.fit(train_scaled, train_t)

def printScore() :    
    # train_scaled 점수 출력
    print(sc.score(train_scaled, train_t))

    # test_scaled 점수 출력
    print(sc.score(test_scaled, test_t))

x = []
y1 = []
y2 = []

for i in range(int(loop_count / loop_range)) :
    x.append(i * loop_range)
    y1.append(sc.score(train_scaled, train_t))
    y2.append(sc.score(test_scaled, test_t))

    printScore()

    # 마저 학습
    for _ in range(loop_range) :
        sc.partial_fit(train_scaled, train_t)

plt.plot(x, y1)
plt.plot(x, y2)
plt.show()