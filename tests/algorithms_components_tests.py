# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 17:55:27 2021

@author: paana
"""

import time
import HIDIV.visualizations as viz

from sklearn.datasets import make_blobs
from HIDIV.dePDDP import dePDDP

# Example data creation
X, y = make_blobs(n_samples=1500, centers=10, cluster_std=1.2, random_state=41197)
print('Example data shape: ' + str(X.shape))


tic = time.perf_counter()
depddp = dePDDP(max_clusters_number = 7, split_data_bandwidth_scale = 0.5, percentile = 0.2).fit_predict(X)
toc = time.perf_counter()
print(toc-tic)

out_mat = depddp.output_matrix
depddp.cluster_labels


viz.split_viz_on_2_PCs(depddp, color_map='tab20').show()

viz.split_viz_with_density_margin(depddp, color_map='tab20').show()

viz.tree_text_viz(depddp, name="text_tree_viz", format="png", title="Dummy example text info").view()

viz.tree_data_viz(depddp, name="data_tree_viz", title="Dummy example data splits", format="png", color_map="tab20", height=None, width=None, rootLabelActive=False, nodeLabels=True).view()

import HIDIV.inteactive_visualization as iv
iv.main(depddp)