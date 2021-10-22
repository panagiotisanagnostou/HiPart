# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 11:57:37 2021

@author: paana
"""

import math
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from KDEpy import FFTKDE
from scipy.cluster.hierarchy import linkage, dendrogram


import graphviz as gv
from collections import Counter
from tempfile import NamedTemporaryFile


def split_viz_with_density_margin(dePDDP_object, color_map = "tab20"):
    """
    Execute the steps of the algorithm dePDDP untill one of the two stopping creterion is not true.
    
    Parameters
    ----------
        dePDDP_object : dePDDP class object
            The object that we want to manipulated on the premiss of this function
        
    Returns
    -------
        tree : treelib.Tree object
            The newlly created tree after the execution of the algorithm
    """
    # prepare the necessary data for this visualization
    dictionary_of_nodes, internal_nodes, _, sample_color = visualization_preparation(dePDDP_object, color_map)
    number_of_splits = len(internal_nodes)
    
    # insure that the root node will be always an internal node while is is the only node in the tree
    if number_of_splits == 0: internal_nodes = [0]
    
    # plot subfigure
    row_plots = 3 if math.ceil(number_of_splits) > 4 else 2
    
    # set figure size
    fig = plt.figure(figsize=(4 * row_plots, 3.7 * math.ceil(number_of_splits / row_plots)), constrained_layout=True)
    
    # create grid for subfigures
    gs = gridspec.GridSpec(math.ceil(number_of_splits / row_plots) * 4, row_plots * 4, fig, hspace=.04)
    
    for i in range(number_of_splits):
        # exrtact the subplot position
        row_from, row_to, col_from, col_to = grid_position(current = i, rows = row_plots, splits = number_of_splits, with_hist = True)
        
        pr_col = sample_color[dePDDP_object.tree.get_node(internal_nodes[i]).data['indices']]
        principal_projections = dictionary_of_nodes[internal_nodes[i]].data['projection']
        splitPoint = dictionary_of_nodes[internal_nodes[i]].data['splitpoint']
        
        bandwidth = sm.nonparametric.bandwidths.select_bandwidth(principal_projections[:,0], "silverman", kernel=None)
        s, e = FFTKDE(kernel='gaussian', bw = (dePDDP_object.split_data_bandwidth_scale * bandwidth)).fit(principal_projections[:,0]).evaluate()
        
        # create the subplot on a pyplot axes
        scatter = plt.subplot(gs[row_from+1:row_to, col_from:col_to])
        scatter.scatter(principal_projections[:,0], principal_projections[:,1], c=pr_col, s=18, marker=".")
        scatter.axvline(x=splitPoint, color='red', lw=1)
        scatter.set_xticks([])
        scatter.set_yticks([])
        scatter.grid()
        scatter.xaxis.grid(which='minor')
        
        hist = plt.subplot(gs[row_from:row_to-3, col_from:col_to])
        hist.plot(s, e)
        # hist.axvline(x=splitPoint, color='red', lw=1)
        hist.set_xticks([])
        hist.set_yticks([])
        hist.grid()
        hist.autoscale_view()
        
        hist.title.set_text("Original data with 1st split") if i == 0 else hist.title.set_text("Split no. " + str(i+1))
        
    fig.tight_layout()
    return fig


# def dendrogram_visualization(dePDDP_object, color_map = "tab20"):
#     """
#     Execute the steps of the algorithm dePDDP untill one of the two stopping creterion is not true.
    
#     Parameters
#     ----------
#         dePDDP_object : dePDDP class object
#             The object that we want to manipulated on the premiss of this function
        
#     Returns
#     -------
#         tree : treelib.Tree object
#             The newlly created tree after the execution of the algorithm
#     """
    
#     dictionary_of_nodes = dePDDP_object.tree.nodes
    
#     # get colormap
#     color_map = matplotlib.cm.get_cmap(color_map, len(list(dictionary_of_nodes.keys())))
#     color_list = [ color_map(i) for i in range(color_map.N) ]
    
#     # create a dictionary between the color (key) and the identifier (value) of the clusters
#     cluster_key = { i.data['color_key']: i.identifier for i in dePDDP_object.tree.leaves() }
    
#     # extract the path to leaves (clusters) so the distance between the clusters on the tree can be assessed
#     path_to_leaves = dePDDP_object.tree.paths_to_leaves()
    
#     tree_depth = dePDDP_object.tree.depth()
    
#     distance_matrix = np.array([ [ calculate_distance(i, j, cluster_key, path_to_leaves, tree_depth) for j in range(len(cluster_key)) ] for i in range(len(cluster_key)) ])
    
#     inv_cluster_key = { v: k for k, v in cluster_key.items() }
    
#     cluster_labels = dePDDP_object.cluster_labels
    
#     pdist_matrix = [ distance_matrix[ inv_cluster_key[cluster_labels[i]], inv_cluster_key[cluster_labels[j]] ] for i in range(len(cluster_labels)-1) for j in range(i+1, len(cluster_labels))  ]
    
#     linkage_matrix = linkage(pdist_matrix, method="average")
    
#     # set figure size
#     fig = plt.figure(figsize=(10, 4), constrained_layout=True)
    
#     dendrogram(linkage_matrix, color_threshold=1, show_leaf_counts=True)
    
#     return fig


# def calculate_distance(cluster_1, cluster_2, cluster_key, path_to_leaves, tree_depth):
#     if cluster_1 == cluster_2:
#         return 0
    
#     path_1 = which_path(cluster_key[cluster_1], path_to_leaves)
#     path_2 = which_path(cluster_key[cluster_2], path_to_leaves)
    
    
#     for i in range(-1, -len(path_1)-1, -1):
#         if path_1[i] in path_2:
#             common_ancestor = i + len(path_1) + 1
#             break
    
#     distance = (len(path_1) - common_ancestor) + (len(path_2) - common_ancestor) + (common_ancestor*(tree_depth - common_ancestor)/2)
    
#     return distance


# def which_path(cluster, path_to_leaves):
#     for i in path_to_leaves:
#         if cluster == i[-1]:
#             cluster_path = i
    
#     return cluster_path


def visualization_preparation(dePDDP_object, color_map):
    dictionary_of_nodes = dePDDP_object.tree.nodes
    
    # get colormap
    color_map = matplotlib.cm.get_cmap(color_map, len(list(dictionary_of_nodes.keys())))
    color_list = [ color_map(i) for i in range(color_map.N) ]
    
    # find the clusters in the data
    clusters = dePDDP_object.tree.leaves()
    clusters = sorted(clusters, key = lambda x:x.identifier)
    
    # create colormap for the generated clusters
    cluster_map = np.zeros(dePDDP_object.samples_number)
    for i in clusters:
        cluster_map[i.data['indices']] = i.identifier
    # color assignment
    sample_color = np.array([ color_list[int(i)] for i in cluster_map ])
    
    # find all the spliting points of the dataset via the internal nodes of the tree
    number_of_nodes = len(list(dictionary_of_nodes.keys()))
    leaf_node_list = [ j.identifier for j in clusters ]
    internal_nodes = [ i for i in range(number_of_nodes) if not (i in leaf_node_list) ]
    
    return dictionary_of_nodes, internal_nodes, color_list, sample_color


def grid_position(current, rows, splits, with_hist=True):
    curRow = math.floor(current / rows)
    curCol = current % rows
    
    num_of_subgrid_elements = 4 if with_hist else 2
    
    if curRow != math.ceil(splits / rows)-1:
            row_from = (curRow * num_of_subgrid_elements)
            row_to = (curRow * num_of_subgrid_elements) + num_of_subgrid_elements
            col_from = (curCol * num_of_subgrid_elements)
            col_to = (curCol * num_of_subgrid_elements) + num_of_subgrid_elements
    else:
        if splits % rows == 0:
            row_from = (curRow * num_of_subgrid_elements)
            row_to = (curRow * num_of_subgrid_elements) + num_of_subgrid_elements
            col_from = (curCol * num_of_subgrid_elements)
            col_to = (curCol * num_of_subgrid_elements) + num_of_subgrid_elements
            
        elif splits % rows != 0:
            position_corection = ((rows - 1) / (splits % rows))
            row_from = (curRow * num_of_subgrid_elements)
            row_to = (curRow * num_of_subgrid_elements) + num_of_subgrid_elements
            col_from = int((curCol * num_of_subgrid_elements) + position_corection)
            col_to = int((curCol * num_of_subgrid_elements) + position_corection) + num_of_subgrid_elements
        
    return row_from, row_to, col_from, col_to





