# -*- coding: utf-8 -*-
"""
Paper dendrogram figure generation.

@author: Panagiotis Anagnostou
"""

from HiPart.clustering import dePDDP
from scipy.cluster import hierarchy

import h5py
import HiPart.__utility_functions as util
import HiPart.visualizations as viz
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def h5file(data_folder, name):
    print("\n", name)

    f = h5py.File(data_folder + name + ".h5", "r")
    inData = f["data"]["matrix"][:].transpose()
    inTarget = f["class"]["categories"][:]
    inTarget = np.int32(inTarget) - 1

    if inData.shape[0] != len(inTarget):
        inData = inData.transpose()
        if inData.shape[0] != len(inTarget):
            print("Data ", name, "error! Pls Check!")
            f.close()
            return
    f.close()

    return inData, inTarget


if __name__ == "__main__":
    # laod the data
    X = pd.read_csv("./data/cancer/data.csv", index_col=0, header=0)
    X = np.asarray(X, dtype="float64")
    y = pd.read_csv("./data/cancer/labels.csv", index_col=0, header=0)
    y["Class"] = pd.Categorical(y["Class"])
    y["Class"] = y.Class.cat.codes
    y = np.asarray(y).transpose()[0]

    print("\ncancer")
    print(X.shape)

    # execution of the dePDDP algorithm
    depddp = dePDDP(
        decomposition_method="pca",
        max_clusters_number=np.unique(y).shape[0],
        bandwidth_scale=0.5,
        percentile=0.1,
    ).fit(X)

    # color map
    color_map = matplotlib.cm.get_cmap("tab20", 14)
    color_list = [util.convert_to_hex(color_map(i)) for i in range(color_map.N)]

    # initialize pyplot rcParams
    plt.rcParams["figure.figsize"] = [4, 4.5]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams['ytick.labelsize'] = 9

    # make figure
    fig = plt.figure(figsize=(5, 7))
    # create a grid
    gs = gridspec.GridSpec(25, 1, fig, wspace=0.01, hspace=.2)

    # dendrogram subplot
    dendro = plt.subplot(gs[0:24, 0:1])
    hierarchy.set_link_color_palette(color_list)
    den_data = viz.dendrogram_visualization(
        depddp,
        count_sort=True,
        no_labels=True,
        above_threshold_color='black',
        ax=dendro
    )
    dendro.set_xticks([])
    dendro.set_yticks([])
    dendro.grid()
    dendro.axis("off")
    dendro.xaxis.grid(which="minor")
    dendro.get_xticks()

    # color the pyrity line
    color_book = {
        0: color_list[0],
        1: color_list[1],
        2: color_list[2],
        3: color_list[3],
        4: color_list[4],
        5: color_list[5],
        6: color_list[6],
        7: color_list[7],
        8: color_list[8],
        9: color_list[9],
        10: color_list[10],
        11: color_list[11],
        12: color_list[12],
    }
    colors = y[den_data["leaves"]]
    colors = np.array([color_book[i] for i in colors])

    # create the purity line
    labels = plt.subplot(gs[24:26, 0:1])
    labels.scatter(
        np.arange(X.shape[0]),
        np.zeros(X.shape[0]),
        s=65,
        c=colors,
        marker="|",
    )
    labels.axis([0, X.shape[0], -0.05, 0.05])
    labels.set_xticks([])
    labels.set_yticks([])
    labels.grid()
    labels.axis("off")
    labels.xaxis.grid(which="minor")

    # save figure
    plt.savefig("dendrogram.png")
