import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from clustering.GMM import *
import matplotlib.pyplot as plt

print("\n\nDADOS GERADOS")

X_1 = np.random.multivariate_normal(mean=[3, 0], cov=[[1, 0], [0, 1]], size=75)
X_2 = np.random.multivariate_normal(mean=[0, 2.5], cov=[[2, 0], [0, 2]], size=250)
X_3 = np.random.multivariate_normal(mean=[0, -2.5], cov=[[1, 0], [0, 2]], size=30)
df = np.concatenate([X_1, X_2, X_3])

km = GMM(k=3)
km.fit(df)
labels = km.predict(df)
print("Predição final")
print(labels)
centroids = km.medias

fig, ax = plt.subplots(1, 3, figsize=(10, 10))
ax[0].scatter(X_1[:, 0], X_1[:, 1])
ax[0].scatter(X_2[:, 0], X_2[:, 1])
ax[0].scatter(X_3[:, 0], X_3[:, 1])
ax[0].set_aspect('equal')
ax[0].set_title("Original")

labels = km.predictKMeans(df)
centroids = km.centers
ax[1].scatter(df[:, 0], df[:, 1], c=labels)
ax[1].scatter(centroids[:, 0], centroids[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')
for i, c in enumerate(centroids):
    ax[1].scatter(c[0], c[1], marker='$%d$' % i, s=50, alpha=1, edgecolor='r')
ax[1].set_aspect('equal')
ax[1].set_title("KMeans")

labels = km.predict(df)
centroids = km.medias
ax[2].scatter(df[:, 0], df[:, 1], c=labels)
ax[2].scatter(centroids[:, 0], centroids[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')
for i, c in enumerate(centroids):
    ax[2].scatter(c[0], c[1], marker='$%d$' % i, s=50, alpha=1, edgecolor='r')
ax[2].set_aspect('equal')
ax[2].set_title("GMM")

fig.suptitle("Clusteres originais e comparação de GMM e KMeans")
plt.tight_layout()
plt.show()
