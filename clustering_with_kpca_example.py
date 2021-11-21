# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 11:28:40 2021

@author: Panagiotis Anagnostou
"""

import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles

np.random.seed(0)

X, y = make_circles(n_samples=400, factor=.3, noise=.05)

kPCA_params = {'kernel':"rbf", 'fit_inverse_transform':True, 'gamma':1.5}

kpca = KernelPCA(**kPCA_params)
X_kpca = kpca.fit_transform(X)
X_back = kpca.inverse_transform(X_kpca)
pca = PCA()
X_pca = pca.fit_transform(X)

plt.title("Original space")
reds = y == 0
blues = y == 1

plt.scatter(X[reds, 0], X[reds, 1], c="red", s=20, edgecolor='k')
plt.scatter(X[blues, 0], X[blues, 1], c="blue", s=20, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

X1, X2 = np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))
X_grid = np.array([np.ravel(X1), np.ravel(X2)]).T
# projection on the first principal component (in the phi space)
Z_grid = kpca.transform(X_grid)[:, 0].reshape(X1.shape)
plt.contour(X1, X2, Z_grid, colors='grey', linewidths=1, origin='lower')


from HIDIV.dePDDP import dePDDP

clusternumber=2

tic = time.perf_counter()
outObj = dePDDP(decomposition_method='kpca' , max_clusters_number = clusternumber, bandwidth_scale = 0.5, percentile = 0.1, **kPCA_params).fit(X)
toc = time.perf_counter()
print(toc-tic)

import HIDIV.visualizations as viz
viz.split_visualization(outObj).show()
out_y = outObj.labels_

plt.title("Clusterd data")
reds = out_y == 1
blues = out_y == 2

plt.scatter(X[reds, 0], X[reds, 1], c="red", s=20, edgecolor='k')
plt.scatter(X[blues, 0], X[blues, 1], c="blue", s=20, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

X1, X2 = np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))
X_grid = np.array([np.ravel(X1), np.ravel(X2)]).T
# projection on the first principal component (in the phi space)
Z_grid = kpca.transform(X_grid)[:, 0].reshape(X1.shape)
plt.contour(X1, X2, Z_grid, colors='grey', linewidths=1, origin='lower')

