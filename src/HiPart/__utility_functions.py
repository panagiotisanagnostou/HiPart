# -*- coding: utf-8 -*-
"""
Utility fuctions of the HiPart package.

@author: Panagiotis Anagnostou
"""

import numpy as np
import pandas as pd
import pickle
import warnings

from sklearn.decomposition import PCA, KernelPCA, FastICA


def execute_decomposition_method(data_matrix, decomposition_method, decomposition_args):
    """
    Projection of the data matrix onto its first two Components with the
    utilization of the "Principal Components Analysis", "Kernel Principal
    Components Analysis" or "Independent Component Analysis" decomposition
    methods.

    Parameters
    ----------
    data_matrix : numpy.ndarray
        The data matrix contains all the data for the samples.
    decomposition_method : str
        One of 'kpca', 'pca' and 'ica' the decomposition methods supported by
        this software.
    decomposition_args : dict
        Arguments to use by each of the decomposition methods utilized by the
        HIDIV package.

    Returns
    -------
    two_dimensions : numpy.ndarray
        The projections of the samples on the first two components of the pca
        and kernel pca methods.

    """

    if decomposition_method == "kpca":
        kernel_pca = KernelPCA(n_components=2, **decomposition_args)
        two_dimensions = kernel_pca.fit_transform(data_matrix)
    elif decomposition_method == "pca":
        pca = PCA(n_components=2, **decomposition_args)
        two_dimensions = pca.fit_transform(data_matrix)
    elif decomposition_method == "ica":
        ica = FastICA(n_components=2, **decomposition_args)
        two_dimensions = ica.fit_transform(data_matrix)
    else:
        raise ValueError(
            ": The dicomposition method ("
            + decomposition_method
            + ") is not supported!"
        )

    return two_dimensions


def center_data(data):
    """
    Center the data on all its dimensions (subtract the mean of each variable,
    from each variable).

    Parameters
    ----------
    data : numpy.ndarray
        The data matrix containing all the data for the samples, samples are
        the rows and variables are the columns.

    Returns
    -------
    centered : numpy.ndarray
        The input data matrix centered on its variables.

    """

    # calculation of the mean of each variable (column)
    mean = np.nanmean(data, axis=0)
    # Subtract the mean from each sample of the variable, for each variable
    # separately.
    centered = data - mean

    mean_1 = np.nanmean(centered, axis=0)
    # Verify that mean_1 is 'close to zero'. If X contains very large values,
    # mean_1 can also be very large, due to a lack of precision of mean_. In
    # this case, a pre-scaling of the concerned feature is efficient, for
    # instance by its mean or maximum.
    if not np.allclose(mean_1, 0):
        warnings.warn(
            """Numerical issues were encountered when centering the data and
            might not be solved. Dataset may contain too large values. You may
            need to prescale your features."""
        )
        centered -= mean_1

    return centered


def data_preparation(object_path, splitVal):
    """
    Generate the necessary data for all the visualizations.

    Parameters
    ----------
    object_path : str
        The location of the temporary file containg the pickel dump of the
        object we want to visualize.
    splitVal : int
        The serial number of split that want to extract data from.

    Returns
    -------
    data_matrix : pandas.core.frame.DataFrame
    splitpoint : int
    internal_nodes : list
    number_of_nodes : int

    """

    # load the data from the temp files
    with open(object_path, "rb") as obj_file:
        tree = pickle.load(obj_file).tree

    # Find the clusters from the algorithms tree
    clusters = tree.leaves()
    clusters = sorted(clusters, key=lambda x: x.identifier)

    root = tree.get_node(0)

    # match the points to their respective cluster
    cluster_map = np.zeros(len(root.data["indices"]))
    for i in clusters:
        cluster_map[i.data["indices"]] = str(int(i.data["color_key"]))

    # list of all the tree's nodes
    dictionary_of_nodes = tree.nodes

    # Search for the internal nodes (splits) of the tree with the use of the
    # clusters determined above
    number_of_nodes = len(list(dictionary_of_nodes.keys()))
    leaf_node_list = [j.identifier for j in clusters]
    internal_nodes = [i for i in range(number_of_nodes) if not (i in leaf_node_list)]

    # When nothing din the tree but the root insert the root as internal
    # node (tree theory)
    if len(internal_nodes) == 0:
        internal_nodes = [0]

    # based on the splitVal imported find the node to split
    node_to_visualize = (
        tree.get_node(internal_nodes[splitVal])
        if len(internal_nodes) != 0
        else tree.get_node(0)
    )

    # create a data matix containg the 1st and 2nd principal components and
    # each clusters respective color key
    data_matrix = pd.DataFrame(
        {
            "PC1": node_to_visualize.data["projection"][:, 0],
            "PC2": node_to_visualize.data["projection"][:, 1],
            "cluster": cluster_map[node_to_visualize.data["indices"]],
        }
    )

    data_matrix["cluster"] = data_matrix["cluster"].astype(int).astype(str)

    # determine the splitpoint value
    splitpoint = node_to_visualize.data["splitpoint"]

    return data_matrix, splitpoint, internal_nodes, number_of_nodes


def convert_to_hex(rgba_color):
    """
    Conver the color enconding from RGBa to hexadecimal for integration with
    the CSS.

    Parameters
    ----------
    rgba_color : tuple
        A tuple of floats containing the RGBa values.

    Returns
    -------
    str
        The hexadecimal value of the color in question.

    """

    red = int(rgba_color[0] * 255)
    green = int(rgba_color[1] * 255)
    blue = int(rgba_color[2] * 255)

    return "#{r:02x}{g:02x}{b:02x}".format(r=red, g=green, b=blue)


def message_center(message_id, object_path):
    """


    Parameters
    ----------
    object_path : TYPE
        DESCRIPTION.
    message_id : TYPE
        DESCRIPTION.

    Returns
    -------
    message : str
        The requested message.

    """

    with open(object_path, "rb") as obj_file:
        obj = pickle.load(obj_file)

    obj_name = str(obj.__class__).split(".")[-1].split("'")[0]

    if message_id == "des:main_cluser":
        msg = """
        ### """ + obj_name + """: Basic viualization

        This is the basic visualization for the clustering of the input data.
        This figure is generated by visualizing the first two components of the
        decomposition method used by the """ + obj_name + """ algorithm. The
        colors on visualization represent each of the separate clusters
        extracted by the execution of the """ + obj_name + """ algorithm. It is
        important to notice that each color represents the same cluster
        throughout the execution of the interactive visualization (the colors
        on the clusters only change with the manipulation of the execution of
        the algorithm).

        """

    elif message_id == "des:splitpoitn_man":
        msg = """
        ### """ + obj_name + """: Splitpoint Manipulation

        On this page, you can manipulate the split point of the
        """ + obj_name + """ algorithm. The process is similar to all the
        algorithms, members of the HiPart package.

        For the split-point manipulation, the top section of the figure is the
        selection of split to manipulate. This can be done with the help of the
        top sliding bar. The numbers below the bar each time represent the
        serial number of the split to manipulate. Zero is the first split of
        the data set. It is necessary to notice that the manipulation of the
        split point because of the nature of the execution must start from the
        earliest to the latest split otherwise, the manipulation will be lost.

        The middle section of the figure visualizes the data with the
        utilization of the decomposition technique used to execute the
        """ + obj_name + """ algorithm. The colors on the scatter plot
        represent the final clusters of the input dataset. The red vertical
        line represents the split-point of the data for the current split of
        the algorithm. """

        if obj_name == "dePDDP":
            msg += """The marginal plot for the x-axis of this visualization
            represents the density of the data. The reason behind that is to
            visualize the information the algorithm has to split the data.

            """
        else:
            msg += """The marginal plot for the x-axis of this visualization is
            one dimension scatter plot of the split, the data of which we
            visualize. The reason behind that is to visualize the information
            the algorithm has to split the data.

            """
        msg += """
        Finally, the bottom section of the figure allows the manipulation of
        the split-point with the utilization of a continuous sliding bar. By
        changing positions on the node of the sliding-bar we can see
        corresponding movements on the vertical red line that represents the
        split-point of the split.

        The apply button, at the bottom of the page, applies the manipulated
        split-point and re-executes the algorithm for the rest of the splits
        that appeared after the currently selected one.
        """

    return msg
