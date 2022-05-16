# -*- coding: utf-8 -*-
"""
mlkia GAMW
"""
import matplotlib.pyplot as plt
import time
import HiPart.inteactive_visualization as inteactive_visualization
import HiPart.visualizations as viz

from HiPart.clustering import bicecting_kmeans
from HiPart.clustering import dePDDP
from HiPart.clustering import iPDDP
from HiPart.clustering import kM_PDDP
from HiPart.clustering import PDDP
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi

# %% Example data creation
clusters = 6

X, y = make_blobs(
    n_samples=1500,
    centers=8,
    cluster_std=0.5,
    random_state=41179,
)
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

print("depddp_time= {val:.5f}".format(val=toc-tic))
print("depddp_mni= {val:.5f}".format(val=nmi(y, depddp.labels_)))
print("depddp_ari= {val:.5f}\n".format(val=ari(y, depddp.labels_)))

# scatter visualization
viz.split_visualization(depddp).show()
# dendrogram
plt.figure(figsize=(10, 3))
dn = viz.dendrogram_visualization(depddp)
plt.show()

# interactive visualization
inteactive_visualization.main(depddp)


# %% Bisecting kMeans algorithm execution
tic = time.perf_counter()
bikmeans = bicecting_kmeans(max_clusters_number=clusters).fit(X)
toc = time.perf_counter()

print("bikmeans_time= {val:.5f}".format(val=toc-tic))
print("bikmeans_mni= {val:.5f}".format(val=nmi(y, bikmeans.labels_)))
print("bikmeans_ari= {val:.5f}\n".format(val=ari(y, bikmeans.labels_)))

# # scatter visualization
# viz.split_visualization(bikmeans).show()
# # dendrogram
# plt.figure(figsize=(10, 3))
# viz.dendrogram_visualization(bikmeans)
# # bisecting kMeans is not supported by the interactive visualization


# %% kMeans-PDDP algorithm execution
tic = time.perf_counter()
kmpddp = kM_PDDP(
    decomposition_method="pca",
    max_clusters_number=clusters,
).fit(X)
toc = time.perf_counter()

print("kmpddp_time= {val:.5f}".format(val=toc-tic))
print("kmpddp_mni= {val:.5f}".format(val=nmi(y, kmpddp.labels_)))
print("kmpddp_ari= {val:.5f}\n".format(val=ari(y, kmpddp.labels_)))

# # scatter visualization
# viz.split_visualization(kmpddp).show()
# # dendrogram
# plt.figure(figsize=(10, 3))
# viz.dendrogram_visualization(kmpddp)
# # interactive visualization
# inteactive_visualization.main(kmpddp)


# %% PDDP algorithm execution
tic = time.perf_counter()
pddp = PDDP(decomposition_method="pca", max_clusters_number=clusters).fit(X)
toc = time.perf_counter()

print("pddp_time= {val:.5f}".format(val=toc-tic))
print("pddp_mni= {val:.5f}".format(val=nmi(y, pddp.labels_)))
print("pddp_ari= {val:.5f}\n".format(val=ari(y, pddp.labels_)))

# # scatter visualization
# viz.split_visualization(pddp).show()
# # dendrogram
# plt.figure(figsize=(10, 3))
# viz.dendrogram_visualization(pddp)
# # interactive visualization
# inteactive_visualization.main(pddp)


# %% iPDDP algorithm execution
tic = time.perf_counter()
ipddp = iPDDP(
    decomposition_method="pca", max_clusters_number=clusters, percentile=0.1
).fit(X)
toc = time.perf_counter()

print("ipddp_time= {val:.5f}".format(val=toc-tic))
print("ipddp_mni= {val:.5f}".format(val=nmi(y, ipddp.labels_)))
print("ipddp_ari= {val:.5f}\n".format(val=ari(y, ipddp.labels_)))

# # scatter visualization
# viz.split_visualization(ipddp).show()
# # dendrogram
# plt.figure(figsize=(10, 3))
# viz.dendrogram_visualization(ipddp)
# # interactive visualization
# inteactive_visualization.main(ipddp)
