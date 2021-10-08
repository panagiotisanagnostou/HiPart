import time

from sklearn.datasets import make_blobs
from src.dePDDPcl import dePDDP

# Example data creation
X, y = make_blobs(n_samples=1500, centers=10, cluster_std=1.2, random_state=41197)
print('Example data shape: ' + str(X.shape))


tic = time.perf_counter()
depddp = dePDDP(max_clusters_number = 7, split_data_bandwidth_scale = 0.5, percentile = 0.2).fit_predict(X)
toc = time.perf_counter()
print(toc-tic)

out_mat = depddp.output_matrix
depddp.cluster_labels


depddp.split_viz_on_2_PCs(color_map='tab20').show()

depddp.split_viz_with_density_margin(color_map='tab20').show()

depddp.tree_text_viz(name="text_tree_viz", title="Dummy example text info")

depddp.tree_data_viz(name="data_tree_viz", title="Dummy example data splits", format="png", color_map="tab20", height=None, width=None, rootLabelActive=False, nodeLabels=True).view()

import src.inteactive_visualization as iv
iv.main(depddp)
