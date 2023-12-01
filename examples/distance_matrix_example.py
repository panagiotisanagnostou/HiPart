from HiPart.clustering import DePDDP
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
from scipy.spatial import distance_matrix

import HiPart.visualizations as viz
import HiPart.interactive_visualization as iviz
import time

# %% Example data creation
# number of cluster in the data
clusters = 5

X, y = make_blobs(
    n_samples=500,
    centers=5,
    cluster_std=.8,
    random_state=123,
)
print("Example data shape: {}\n".format(X.shape))

# Calculate distance matrix
dist_matrix = distance_matrix(X, X)

# %% dePDDP algorithm execution
# timer for the execution time in the form of tic-toc
tic = time.perf_counter()
depddp = DePDDP(
    decomposition_method="mds",
    max_clusters_number=clusters,
    bandwidth_scale=0.5,
    percentile=0.1,
    distance_matrix=True,
    random_state=12,
).fit(dist_matrix)
toc = time.perf_counter()

# results evaluation in terms of execution time, MNI and ARI metrics
print("depddp_time= {val:.5f}".format(val=toc - tic))
print("depddp_mni= {val:.5f}".format(val=nmi(y, depddp.labels_)))
print("depddp_ari= {val:.5f}\n".format(val=ari(y, depddp.labels_)))

# scatter visualization
viz.split_visualization(depddp).show()
# interactive visualization
iviz.main(depddp)