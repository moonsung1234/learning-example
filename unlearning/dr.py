
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

fruits = np.load("fruits.npy")
fruits_2d = fruits.reshape(-1, 100 * 100)

lr = LogisticRegression()

pca = PCA(n_components=0.5)
pca.fit(fruits_2d)

fruits_pca = pca.transform(fruits_2d)

target = np.array([0] * 100 + [1] * 100 + [2] * 100)
scores = cross_validate(lr, fruits_pca, target)

km = KMeans(n_clusters=3, random_state=2021)
km.fit(fruits_pca)

for i in range(3) :
    data = fruits_pca[km.labels_ == i]
    
    plt.scatter(data[:, 0], data[:, 1])

plt.legend(["apple", "banana", "pineapple"])
plt.show()

print("score : ", np.mean(scores["test_score"]))
print("fit time : ", np.mean(scores["fit_time"]))