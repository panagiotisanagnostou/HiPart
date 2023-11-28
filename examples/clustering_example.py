from HiPart.clustering import BisectingKmeans
from HiPart.clustering import DePDDP
from HiPart.clustering import IPDDP
from HiPart.clustering import KMPDDP
from HiPart.clustering import PDDP
from HiPart.clustering import MDH
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi

import matplotlib.pyplot as plt
import time
import HiPart.visualizations as viz

# %% Example data creation
# number of cluster in the data
clusters = 5

X, y = make_blobs(
    n_samples=1500,
    centers=5,
    cluster_std=.8,
    random_state=123,
)
print("Example data shape: {}\n".format(X.shape))

# %% dePDDP algorithm execution
# timer for the execution time in the form of tic-toc
tic = time.perf_counter()
depddp = DePDDP(
    decomposition_method="pca",
    max_clusters_number=clusters,
    bandwidth_scale=0.5,
    percentile=0.1,
).fit(X)
toc = time.perf_counter()

# results evaluation in terms of execution time, MNI and ARI metrics
print("depddp_time= {val:.5f}".format(val=toc - tic))
print("depddp_mni= {val:.5f}".format(val=nmi(y, depddp.labels_)))
print("depddp_ari= {val:.5f}\n".format(val=ari(y, depddp.labels_)))

# scatter visualization
viz.split_visualization(depddp).show()
# dendrogram
dn = viz.dendrogram_visualization(depddp)
plt.show()

# %% iPDDP algorithm execution
# timer for the execution time in the form of tic-toc
tic = time.perf_counter()
ipddp = IPDDP(
    decomposition_method="pca", max_clusters_number=clusters, percentile=0.1
).fit(X)
toc = time.perf_counter()

# results evaluation in terms of execution time, MNI and ARI metrics
print("ipddp_time= {val:.5f}".format(val=toc - tic))
print("ipddp_mni= {val:.5f}".format(val=nmi(y, ipddp.labels_)))
print("ipddp_ari= {val:.5f}\n".format(val=ari(y, ipddp.labels_)))

# scatter visualization
viz.split_visualization(ipddp).show()
# dendrogram
dn = viz.dendrogram_visualization(ipddp)
plt.show()

# %% kMeans-PDDP algorithm execution
# timer for the execution time in the form of tic-toc
tic = time.perf_counter()
kmpddp = KMPDDP(
    decomposition_method="pca",
    max_clusters_number=clusters,
).fit(X)
toc = time.perf_counter()

# results evaluation in terms of execution time, MNI and ARI metrics
print("kmpddp_time= {val:.5f}".format(val=toc - tic))
print("kmpddp_mni= {val:.5f}".format(val=nmi(y, kmpddp.labels_)))
print("kmpddp_ari= {val:.5f}\n".format(val=ari(y, kmpddp.labels_)))

# scatter visualization
viz.split_visualization(kmpddp).show()
# dendrogram
dn = viz.dendrogram_visualization(kmpddp)
plt.show()

# %% PDDP algorithm execution
# timer for the execution time in the form of tic-toc
tic = time.perf_counter()
pddp = PDDP(decomposition_method="pca", max_clusters_number=clusters).fit(X)
toc = time.perf_counter()

# results evaluation in terms of execution time, MNI and ARI metrics
print("pddp_time= {val:.5f}".format(val=toc - tic))
print("pddp_mni= {val:.5f}".format(val=nmi(y, pddp.labels_)))
print("pddp_ari= {val:.5f}\n".format(val=ari(y, pddp.labels_)))

# scatter visualization
viz.split_visualization(pddp).show()
# dendrogram
dn = viz.dendrogram_visualization(pddp)
plt.show()

# %% Bisecting kMeans algorithm execution
# timer for the execution time in the form of tic-toc
tic = time.perf_counter()
bikmeans = BisectingKmeans(max_clusters_number=clusters).fit(X)
toc = time.perf_counter()

print("bikmeans_time= {val:.5f}".format(val=toc - tic))
print("bikmeans_mni= {val:.5f}".format(val=nmi(y, bikmeans.labels_)))
print("bikmeans_ari= {val:.5f}\n".format(val=ari(y, bikmeans.labels_)))

# scatter visualization
viz.split_visualization(bikmeans).show()
# dendrogram
dn = viz.dendrogram_visualization(bikmeans)
plt.show()

# %% MDH algorithm execution
# timer for the execution time in the form of tic-toc
tic = time.perf_counter()
mdh = MDH(max_clusters_number=clusters).fit(X)
toc = time.perf_counter()

print("mdh_time= {val:.5f}".format(val=toc - tic))
print("mdh_mni= {val:.5f}".format(val=nmi(y, mdh.labels_)))
print("mdh_ari= {val:.5f}\n".format(val=ari(y, mdh.labels_)))

# scatter visualization
viz.split_visualization(mdh).show()
# dendrogram
dn = viz.dendrogram_visualization(mdh)
plt.show()
