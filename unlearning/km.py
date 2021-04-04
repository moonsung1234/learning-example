
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

fruits = np.load("fruits.npy")
fruits_2d = fruits.reshape(-1, 100 * 100)

km = KMeans(n_clusters=3, random_state=2021)
km.fit(fruits_2d)

# print(km.labels_)
# print(np.unique(km.labels_, return_counts=True))

def draw(arr, ratio=1) :
    n = len(arr)
    rows = int(np.ceil(n / 10))
    cols = n if rows == 1 else 10
    fig, axs = plt.subplots(rows, cols, figsize=(rows * cols, rows * ratio), squeeze=False)

    print("n : ", n)
    print("row : ", rows)
    print("col : ", cols)

    for i in range(rows) :
        for j in range(cols) :
            if i * 10 + j < n : 
                axs[i, j].imshow(arr[i * 10 + j], cmap="gray_r")

            axs[i, j].axis("off")

    plt.show()

# draw(fruits[km.labels_==0])
# draw(fruits[km.labels_==1])
# draw(fruits[km.labels_==2])
# draw(km.cluster_centers_.reshape(-1, 100, 100))

# print(km.transform(fruits_2d[100:101]))
# print(km.predict(fruits_2d[100:101]))
# draw(fruits[100:101])
# print(km.n_iter_)

# 옐보우 방법
iner = []

for k in range(1, 7) :
    km = KMeans(n_clusters=k, random_state=2021)
    km.fit(fruits_2d)

    iner.append(km.inertia_)

plt.plot(range(1, 7), iner)
plt.xlabel("k")
plt.ylabel("inertia")
plt.show()