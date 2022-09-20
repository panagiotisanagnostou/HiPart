from HiPart.clustering import dePDDP
from sklearn.datasets import make_blobs

import HiPart.interactive_visualization as int_viz

# Example data creation
clusters = 5

X, y = make_blobs(
    n_samples=1500,
    centers=5,
    cluster_std=.8,
    random_state=123,
)
print("Example data shape: {}\n".format(X.shape))

# Data clustering
depddp = dePDDP(
    decomposition_method="pca",
    max_clusters_number=clusters,
    bandwidth_scale=0.5,
    percentile=0.1,
).fit(X)

# interactive visualization execution
int_viz.main(depddp)
