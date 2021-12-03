# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import time
import HiPart.inteactive_visualization as inteactive_visualization
import HiPart.visualizations as viz

from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
from HiPart.clustering import bicecting_kmeans
from HiPart.clustering import dePDDP
from HiPart.clustering import iPDDP
from HiPart.clustering import kM_PDDP
from HiPart.clustering import PDDP

clusters = 6
# Example data creation
X, y = make_blobs(n_samples=1500, centers=8, cluster_std=1.2, random_state=41197)
print("Example data shape: {}\n".format(X.shape))

# %% dePDDP algorithm execution
tic = time.perf_counter()
depddp = dePDDP(
    decomposition_method="pca",
    max_clusters_number=clusters,
    bandwidth_scale=0.5,
    percentile=0.1,
).fit(X)
toc = time.perf_counter()
depddp_time = toc - tic
depddp_mni = nmi(y, depddp.labels_)
depddp_ari = ari(y, depddp.labels_)
print("depddp_time= {val:.5f}".format(val=depddp_time))
print("depddp_mni= {val:.5f}".format(val=depddp_mni))
print("depddp_ari= {val:.5f}\n".format(val=depddp_ari))

# scatter visualization
viz.split_visualization(depddp).show()
# dendrogram
figure = plt.figure(figsize=(10, 3))
viz.dendrogram_visualization(depddp)
# interactive visualization
inteactive_visualization.main(depddp)

# %% Bisecting kMeans algorithm execution
tic = time.perf_counter()
bikmeans = bicecting_kmeans(max_clusters_number=clusters).fit(X)
toc = time.perf_counter()
bikmeans_time = toc - tic
bikmeans_mni = nmi(y, bikmeans.labels_)
bikmeans_ari = ari(y, bikmeans.labels_)
print("bikmeans_time= {val:.5f}".format(val=bikmeans_time))
print("bikmeans_mni= {val:.5f}".format(val=bikmeans_mni))
print("bikmeans_ari= {val:.5f}\n".format(val=bikmeans_ari))

# scatter visualization
viz.split_visualization(bikmeans).show()
# dendrogram
figure = plt.figure(figsize=(10, 3))
viz.dendrogram_visualization(bikmeans)

# %% kMeans-PDDP algorithm execution
tic = time.perf_counter()
kmpddp = kM_PDDP(decomposition_method="pca", max_clusters_number=clusters).fit(X)
toc = time.perf_counter()
kmpddp_time = toc - tic
kmpddp_mni = nmi(y, kmpddp.labels_)
kmpddp_ari = ari(y, kmpddp.labels_)
print("kmpddp_time= {val:.5f}".format(val=kmpddp_time))
print("kmpddp_mni= {val:.5f}".format(val=kmpddp_mni))
print("kmpddp_ari= {val:.5f}\n".format(val=kmpddp_ari))

# scatter visualization
viz.split_visualization(kmpddp).show()
# dendrogram
figure = plt.figure(figsize=(10, 3))
viz.dendrogram_visualization(kmpddp)
# interactive visualization
inteactive_visualization.main(kmpddp)

# %% PDDP algorithm execution
tic = time.perf_counter()
pddp = PDDP(decomposition_method="pca", max_clusters_number=clusters).fit(X)
toc = time.perf_counter()
pddp_time = toc - tic
pddp_mni = nmi(y, pddp.labels_)
pddp_ari = ari(y, pddp.labels_)
print("pddp_time= {val:.5f}".format(val=pddp_time))
print("pddp_mni= {val:.5f}".format(val=pddp_mni))
print("pddp_ari= {val:.5f}\n".format(val=pddp_ari))

# scatter visualization
viz.split_visualization(pddp).show()
# dendrogram
figure = plt.figure(figsize=(10, 3))
viz.dendrogram_visualization(pddp)
# interactive visualization
inteactive_visualization.main(pddp)

# %% iPDDP algorithm execution
tic = time.perf_counter()
ipddp = iPDDP(
    decomposition_method="pca", max_clusters_number=clusters, percentile=0.1
).fit(X)
toc = time.perf_counter()
ipddp_time = toc - tic
ipddp_mni = nmi(y, ipddp.labels_)
ipddp_ari = ari(y, ipddp.labels_)
print("ipddp_time= {val:.5f}".format(val=ipddp_time))
print("ipddp_mni= {val:.5f}".format(val=ipddp_mni))
print("ipddp_ari= {val:.5f}\n".format(val=ipddp_ari))

# # scatter visualization
viz.split_visualization(ipddp).show()
# dendrogram
figure = plt.figure(figsize=(10, 3))
viz.dendrogram_visualization(ipddp)
# interactive visualization
inteactive_visualization.main(ipddp)
