
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np

# 빙어와 도미 길이, 무게 데이터
bream_length = np.array([25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0])
bream_weight = np.array([242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0])
smelt_length = np.array([9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0])
smelt_weight = np.array([6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9])

# 길이랑 무게 더하기
bream_all = np.column_stack((bream_length, bream_weight))
smelt_all = np.column_stack((smelt_length, smelt_weight))

# 두개 잇기
x_all = np.concatenate((bream_all, smelt_all))
y_all = np.concatenate((np.ones((35,)), np.zeros((14,))))
temp_data = np.array([[25, 100]])

# 사전 학습
knn = KNeighborsClassifier()
knn.fit(x_all, y_all)

# 데이터 가르기
train_x, test_x, train_y, test_y = train_test_split(x_all, y_all, random_state=2021)

mean = np.mean(train_x, axis=0) # 평균 계산
std = np.std(train_x, axis=0) # 표준편차 계산
train_scaled = (train_x - mean) / std # 학습데이터 표준점수 계산
test_scaled = (test_x - mean) / std # 확인데이터 표준점수 계산
temp_scaled = (temp_data - mean) / std # 임의의 값 표준점수 계산

# 재학습
knn.fit(train_scaled, train_y)

# 정확도 출력하기
print("score : ", knn.score(test_scaled, test_y))

# 값 예측하기
print("predict : ", knn.predict(temp_scaled))

# 가까이있는 데이터 가져오기
distences, indexs = knn.kneighbors(temp_scaled)

# 그래프 나타내기
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(temp_scaled[:,0], temp_scaled[:,1], marker="^")
plt.scatter(train_scaled[indexs, 0], train_scaled[indexs, 1], marker="D")
plt.xlabel("length")
plt.ylabel("weight")
plt.show()
