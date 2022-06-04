# -*- coding: utf-8 -*-
"""
Utility fuctions of the HiPart package.

@author: Panagiotis Anagnostou
"""

import copy
import json
import math
import matplotlib
import numpy as np
import os
import pandas as pd
import pickle
import plotly.subplots as subplots
import plotly.express as px
import plotly.graph_objects as go
import signal
import statsmodels.api as sm
import warnings

from dash import dcc
from dash import html
from KDEpy import FFTKDE
from sklearn.decomposition import PCA, KernelPCA, FastICA


def execute_decomposition_method(
        data_matrix,
        decomposition_method,
        two_dimentions,
        decomposition_args,
):
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
    if two_dimentions:
        n_of_dimentions = 2
    else:
        n_of_dimentions = 1

    if decomposition_method == "pca":
        pca = PCA(
            n_components=n_of_dimentions,
            **decomposition_args
        )
        two_dimensions = pca.fit_transform(data_matrix)
    elif decomposition_method == "kpca":
        kernel_pca = KernelPCA(
            n_components=n_of_dimentions,
            **decomposition_args
        )
        two_dimensions = kernel_pca.fit_transform(data_matrix)
    elif decomposition_method == "ica":
        ica = FastICA(
            n_components=n_of_dimentions,
            **decomposition_args
        )
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
    mean = np.mean(data, axis=0)
    # Subtract the mean from each sample of the variable, for each variable
    # separately.
    centered = data - mean

    mean_1 = np.mean(centered, axis=0)
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


def make_simple_scatter(sp, splitPoint, PP, pr_col, show_split=True):
    """
    Create an Axes plot visualizing the split and data of a cloud of data.

    Parameters
    ----------
    sp : matplotlib.axes.Axes object
        The Axes for the plot to be drawn.
    splitPoint : int
        The values of the point the data split for this plot.
    PP : numpy.ndarray object
        The projection of the data on the first two Principal Components.
    pr_col : numpy.ndarray object
        An array containing the color of each sample as RGBa tuple.
    show_split : bool, optional
        Show the split line in the subplot. The default is True.

    Returns
    -------
    sp : matplotlib.axes.Axes object
        The resulted Axes plot.

    """

    sp.scatter(PP[:, 0], PP[:, 1], c=pr_col, s=18, marker=".")
    sp.set_xticks([])
    sp.set_yticks([])
    sp.grid(False)
    if show_split:
        sp.axvline(x=splitPoint, color="red", lw=1)
    sp.margins(0.03)

    return sp


def make_scatter_n_hist(
        scatter,
        hist,
        PP,
        splitPoint,
        bandwidth_scale,
        pr_col,
        scaler=None
):
    """
    Create an Axes plot visualizing the split and data of a cloud of data. With
    a marginal plot representing the density of data on the x-axis.

    Parameters
    ----------
    scatter : matplotlib.axes.Axes object
        The Axes for the main plot to be drawn.
    hist : matplotlib.axes.Axes object
        The Axes for the x-axis marginal plot to be drawn.
    splitPoint : int
        The values of the point the data are split for this plot.
    PP : numpy.ndarray object
        The projection of the data on the first two Principal Components.
    pr_col : numpy.ndarray object
        An array containing the color of each sample as RGBa tuple.
    show_split : bool, optional
        Show the split line in the subplot. The default is True.

    Returns
    -------
    sp : matplotlib.axes.Axes object
        The resulted Axes plot.

    """
    bandwidth = sm.nonparametric.bandwidths.select_bandwidth(
        PP[:, 0], "silverman", kernel=None
    )
    s, e = (
        FFTKDE(kernel="gaussian", bw=(bandwidth_scale * bandwidth))
        .fit(PP[:, 0])
        .evaluate()
    )

    # create the subplot on a pyplot axes
    scatter.scatter(PP[:, 0], PP[:, 1], c=pr_col, s=18, marker=".")
    scatter.axvline(x=splitPoint, color="red", lw=1)
    scatter.set_xticks([])
    scatter.set_yticks([])
    scatter.grid()
    scatter.xaxis.grid(which="minor")

    if scaler is None:
        hist.plot(s, e)
    else:
        hist.plot(s, e*(PP.shape[0]/scaler))
    hist.axvline(x=splitPoint, color="red", lw=1)
    hist.set_xticks([])
    hist.set_yticks([])
    hist.grid()
    hist.autoscale_view()


def make_scatter_n_marginal_scatter(
        scatter,
        marginal_scatter,
        PP,
        splitPoint,
        pr_col,
        centers=None
):
    """
    Create an Axes plot visualizing the split and data of a cloud of data. With
    a marginal plot representing projections of the data only on the x-axis.

    Parameters
    ----------
    scatter : matplotlib.axes.Axes object
        The Axes for the main plot to be drawn.
    marginal_scatter : matplotlib.axes.Axes object
        The Axes for the x-axis marginal plot to be drawn.
    splitPoint : int
        The values of the point the data are split for this plot.
    PP : numpy.ndarray object
        The projection of the data on the first two Principal Components.
    pr_col : numpy.ndarray object
        An array containing the color of each sample as RGBa tuple.
    centers : numpy.ndarray or None, optional
        Show the split line in the subplot. The default is True.

    Returns
    -------
    sp : matplotlib.axes.Axes object
        The resulted Axes plot.

    """

    # create the subplot on a pyplot axes
    scatter.scatter(PP[:, 0], PP[:, 1], c=pr_col, s=18, marker=".")
    scatter.axvline(x=splitPoint, color="red", lw=1)
    scatter.set_xticks([])
    scatter.set_yticks([])
    scatter.grid()
    scatter.xaxis.grid(which="minor")

    marginal_scatter.scatter(
        PP[:, 0], np.zeros(PP.shape[0]), c=pr_col, s=18, marker="."
    )
    marginal_scatter.axvline(x=splitPoint, color="red", lw=1)
    if centers is not None:
        marginal_scatter.scatter(
            centers,
            np.zeros(2),
            color="black",
            s=50,
            marker="2"
        )
    marginal_scatter.set_xticks([])
    marginal_scatter.set_yticks([])
    marginal_scatter.grid()
    marginal_scatter.autoscale_view()


def visualization_preparation(hipart_object, color_map):
    """
    Generate the data needed for the execution of the visualizations.

    Parameters
    ----------
    hipart_object : dePDDP or iPDDP or kM_PDDP or PDDP object
        The object member of HiPart package that we want to manipulate on the
        premiss of this function.
    color_map : string
        The name of the matplotlib color map to be used for the data
        visualization.

    Returns
    -------
    dictionary_of_nodes : list
        The list of all the nodes created by the splitting of the data that
        were performed by the HiPart member algorithms.
    internal_nodes : list
        The list of internal nodes of the binary tree that were created by the
        splitting of the data performed by the HiPart algorithms.
    color_list : list
        An array containing the colors to be used by the visualization as RGBa
        tuple.
    sample_color : numpy.ndarray
        An array containing the color of each sample as RGBa tuple.

    """

    dictionary_of_nodes = hipart_object.tree.nodes

    # get colormap
    color_map = matplotlib.cm.get_cmap(
        color_map,
        len(list(dictionary_of_nodes.keys()))
    )
    color_list = [color_map(i) for i in range(color_map.N)]

    # find the clusters from the tree generated by the divisive clustering
    # algorithms
    clusters = hipart_object.tree.leaves()
    clusters = sorted(clusters, key=lambda x: x.identifier)

    # create colormap for the generated clusters
    cluster_map = np.zeros(hipart_object.samples_number)
    for i in clusters:
        cluster_map[i.data["indices"]] = i.identifier
    # assign colors to the samples
    sample_color = np.array([color_list[int(i)] for i in cluster_map])

    # find all the spliting points of the dataset via the internal nodes of
    # the tree
    number_of_nodes = len(list(dictionary_of_nodes.keys()))
    leaf_node_list = [j.identifier for j in clusters]
    internal_nodes = [
        i for i in range(number_of_nodes) if not (i in leaf_node_list)
    ]

    return dictionary_of_nodes, internal_nodes, color_list, sample_color


def grid_position(current, rows, splits, with_marginal=True):
    """
    Find the gridspec quordinents for the current subplot.

    Parameters
    ----------
    current : int
        The current number of the subplot to be drawn.
    rows : int
        The number of plots that to be in its row.
    splits : int
        The final number of split created by the algorithm to be visualized.
    with_marginal : bool, optional
        Generate values with x-axis marginal plot. The default is True.

    Returns
    -------
    row_from : int
        Start row of the subplot to be drawn.
    row_to : int
        End row of the subplot to be drawn.
    col_from : int
        Start column of the subplot to be drawn.
    col_to : int
        End column of the subplot to be drawn.

    """

    curRow = math.floor(current / rows)
    curCol = current % rows

    num_of_subgrid_elements = 4 if with_marginal else 2
    sub_grid_size = 2 if with_marginal else 1

    # Change the last row of the visualization to have the subplot always in
    # the middle of the plot
    if curRow != math.ceil(splits / rows) - 1:
        row_from = curRow * num_of_subgrid_elements
        row_to = (
            (curRow * num_of_subgrid_elements) + num_of_subgrid_elements
        )
        col_from = curCol * num_of_subgrid_elements
        col_to = (curCol * num_of_subgrid_elements) + num_of_subgrid_elements
    else:
        if splits % rows == 0:
            row_from = curRow * num_of_subgrid_elements
            row_to = (
                (curRow * num_of_subgrid_elements) + num_of_subgrid_elements
            )
            col_from = curCol * num_of_subgrid_elements
            col_to = (
                (curCol * num_of_subgrid_elements) + num_of_subgrid_elements
            )

        elif splits % rows != 0:
            position_corection = (rows - 1) / (splits % rows) * sub_grid_size
            row_from = curRow * num_of_subgrid_elements
            row_to = (
                (curRow * num_of_subgrid_elements) + num_of_subgrid_elements
            )
            col_from = (
                int((curCol * num_of_subgrid_elements) + position_corection)
            )
            col_to = (
                int((curCol * num_of_subgrid_elements) + position_corection)
                + num_of_subgrid_elements
            )

    return row_from, row_to, col_from, col_to


def _get_node_depth(path_to_leaves, i):
    """
    This function calculates the depth of the node i inside the algorithm`s
    execution tree.

    Parameters
    ----------
    path_to_leaves : list
        List of lists containg all the paths to the leaves of the tree.
    i : TYPE
        The node the path of which we want to find.

    Returns
    -------
    depth : int
        The depth of the node i.

    """

    for path in path_to_leaves:
        if i in path:
            path = np.array(path)
            depth = path[np.where(path <= i)].size
            break
    return depth


def create_linkage(tree_in):
    """
    Create the linkage matrix for the encoding of the divisive clustring tree
    creatred by the member algorithms of the HiPart package.

    Parameters
    ----------
    tree_in : treelib.tree.Tree
        A divisive tree from on of the HiPart algorithm package.

    Returns
    -------
    Z : numpy.ndarray
        The divisive clustering encoded as a linkage matrix.

    """

    tree = copy.deepcopy(tree_in)

    # extract the path to leaves (clusters) so the distance between the
    # clusters on the tree can be assessed
    path_to_leaves = tree.paths_to_leaves()

    # The depth of the tree that we will use
    max_distance = np.max([len(i) for i in path_to_leaves])
    # The total number of samples the data of the tree contain
    samples_number = len(tree.get_node(tree.root).data["indices"])
    # The indicator for the next free node of the linkage tree we are creating
    dendrogram_counts = samples_number

    # Initialize the linkage matrix
    Z = np.array([[0, 0, 0, 0]])
    # Loop through the nodes of the algorithm`s execution tree and do the
    # necessary connections
    # The loop finishes the execution on the node with ID 0 which is always
    # the root of the algorithm execution tree
    for i in range(len(tree.nodes) - 1, -1, -1):
        # If the node is a leaf of the algorithm`s execution tree connect all
        # the samples of the node on same level untile only one node remains
        if tree.get_node(i).is_leaf():
            if not tree.get_node(i).data["dendrogram_check"]:
                # Set all the samples of the included in the node/cluster as
                # unlinked nodes on the dendrogram tree
                tree.get_node(i).data["unlinked_nodes"] = tree.get_node(
                    i
                ).data["indices"]
                # Create the dendrograms subtree and update the algorithm tree
                # node`s data and the index for the next free node
                (
                    cluster_linkage,
                    tree.get_node(i).data,
                    dendrogram_counts,
                ) = linkage_data_maestro(
                    tree.get_node(i),
                    dendrogram_counts, 0.2
                )
                dendrogram_counts += 1
                Z = np.vstack((Z, cluster_linkage))
        else:
            if not tree.get_node(i).data["dendrogram_check"]:
                # Connect the children of the algorithm tree internal node to
                # a new node on the dendrogram tree
                children = tree.children(i)
                Z = np.vstack(
                    [Z, [
                        children[-1].data["unlinked_nodes"][0],
                        children[-2].data["unlinked_nodes"][0],
                        max_distance - _get_node_depth(path_to_leaves, i),
                        children[-1].data["counts"] + children[-2].data["counts"],
                    ]]
                )
                # Update of the data of the algorithm execution tree node
                tree.get_node(
                    i
                ).data["dendromgram_indicator"] = dendrogram_counts
                tree.get_node(i).data["counts"] = (
                    children[-1].data["counts"] + children[-2].data["counts"]
                )
                tree.get_node(i).data["unlinked_nodes"] = [dendrogram_counts]
                dendrogram_counts += 1

    # Remove the first row of the linkage matrix bacause it is the
    # initalixation`s row of zeros
    Z = Z[1:, :]

    return Z


def linkage_data_maestro(node, dendrogram_counts, distance):
    """
    Manages the process of the dendrogram`s subtree creation for a cluster
    extracted from a divisive algorithm of the HiPart package. This process
    includes the indication of the unlinked nodes included in the cluster (the
    leaf node of the divisive algorithm`s tree) for the dendrogram tree.

    Parameters
    ----------
    node : treelib.node.Node
        The algorim tree node we want to create a subtree for the dendrogram
        tree from.
    dendrogram_counts : int
        The next free node of the dendrgram subtree.
    distance : float
        The distance the dendrogram subtree have.

    Returns
    -------
    cluster_linkage : numpy.ndarray
        The linkage matrix of the cluster.
    dict
        The data of the updated algorithm tree`s node data.
    dendrogram_counts : int
        The last unlinked node of the dendrogram`s tree.

    """

    # Update the indicator of the node of the algorithm`s tree which indicates
    # that the node has been added to the dendrogram`s tree
    node.data["dendrogram_check"] = True

    leave = node.data["unlinked_nodes"]
    cluster_linkage, dendrogram_counts = create_cluster_linkage(
        leave, np.ones(len(leave)), dendrogram_counts, distance
    )

    node.data["dendromgram_indicator"] = dendrogram_counts
    node.data["counts"] = cluster_linkage[-1, -1]

    # The only unliked node is the root of the dendrograms subtree
    node.data["unlinked_nodes"] = [dendrogram_counts]

    return cluster_linkage, node.data, dendrogram_counts


def create_cluster_linkage(points, samples_count, node_indicator, distance):
    """
    A recursive function that creates the linkage of the subtree dendrogram
    tree. The execution of the function ends when only one unliked node exists
    in the subtree.

    Parameters
    ----------
    points : numpy.ndarray
        The unliked points in the dendrogram tree.
    samples_count : numpy.ndarray
        The samples count of the unliked points in the dendrogram tree.
    node_indicator : int
        The next free node of the dendrogram subtree.
    distance : float
        The distance the dendrogram subtree has.

    Returns
    -------
    numpy.ndarray
       The linkage matrix of the cluster.
    int
        The last unlinked node of the dendrogram`s tree.

    """

    elements = len(points)

    # End the execution of the recursive function
    if elements == 1:
        return None, node_indicator - 1

    # If the number of the unlinked nodes are odd seperate the last unlinked
    # node and add it afterword as a new subtree
    if (elements % 2) == 0:
        # Create the linkage of the points
        Z, connections = create_the_connections(
            points, samples_count, node_indicator, distance
        )
        counts = Z[:, 3]
    else:
        # Keep the last unlinked node`s data
        forever_alone = points[-1]
        alone_count = samples_count[-1]
        alone_conection = node_indicator

        # Remove the last unlinked node
        points = points[:-1]
        samples_count = samples_count[:-1]

        # Create the linkage of the points
        Z, connections = create_the_connections(
            points, samples_count, node_indicator, distance
        )
        counts = Z[:, 3]

        # Add the last unlinked node`s sub tree
        Z = np.vstack(
            [Z, [
                forever_alone,
                alone_conection,
                distance, Z[0, 3] + alone_count
            ]]
        )
        node_indicator += 1
        counts[0] += alone_count

    # Recursive call for the next level unliked created
    new_nodes = np.array(
        [i for i in range(node_indicator, node_indicator + connections)]
    )
    node_indicator += connections
    next_Z, dendrogram_counts = create_cluster_linkage(
        new_nodes, counts, node_indicator, distance
    )

    # Either return the linkage create with the node indicator for the next
    # free node (end of the recursive call) or merge the previous linkage
    # lower level linkage with the new higher one.
    if next_Z is None:
        return Z, dendrogram_counts
    else:
        return np.vstack((Z, next_Z)), dendrogram_counts


def create_the_connections(points, samples_count, node_indicator, distance):
    """
    Create the linkage of the points.

    Parameters
    ----------
    points : numpy.ndarray
        The unliked points in the dendrogram tree.
    samples_count : numpy.ndarray
        The samples count of the unliked points in the dendrogram tree.
    node_indicator : int
        The next free node of the dendrogram subtree.
    distance : float
        The distance the dendrogram subtree has.

    Returns
    -------
    numpy.ndarray
       The linkage matrix of the cluster.
    int
        The last unlinked node of the dendrogram`s tree.

    """
    elements = len(points)

    left = points[range(0, elements, 2)]
    right = np.delete(points, range(0, elements, 2))

    temp = samples_count[range(0, elements, 2)]
    samples_count = np.delete(samples_count, range(0, elements, 2)) + temp

    distance = np.full(elements // 2, distance)

    return np.column_stack((
        left,
        right,
        distance,
        samples_count,
    )), elements // 2


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
    internal_nodes = [
        i for i in range(number_of_nodes) if not (i in leaf_node_list)
    ]

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
    This function is responsible for saving and returning all the messages sawn
    on the interactive visualization.

    Parameters
    ----------
    object_path : TYPE
        The absolute path of the object the interactive visualization
        visualizes.
    message_id : TYPE
        The id of the messege to be returned.

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


def app_layout(app, tmpFileNames):
    """
    Basic interface creation for the interactive application. The given inputs
    let the user manage the application correctly.

    Parameters
    ----------
    app : dash.Dash
        The application we want to create the layout on.
    tmpFileNames : dict
        A dictionary with the names (paths) of the temporary files needed for
        the execution of the algorithm.

    """

    app.layout = html.Div(
        children=[
            # ----------------------------------------------------------------
            # Head of the interactive visualization
            dcc.Location(id="url", refresh=False),
            html.Div(dcc.Link("X", id="shutdown", href="/shutdown")),
            html.H1("HiPart: Interactive Visualisation"),
            html.Ul(
                children=[
                    html.Li(
                        dcc.Link(
                            "Clustgering results",
                            href="/clustering_results"
                        ),
                        style={"display": "inline", "margin": "0px 5px"},
                    ),
                    html.Li(
                        dcc.Link(
                            "Split Point Manipulation",
                            href="/splitpoint_manipulation"
                        ),
                        style={"display": "inline", "margin": "0px 5px"},
                    ),
                    # html.Li(dcc.Link('Delete Split', href='/delete_split'),
                    #         style={'display': 'inline', 'margin': '0px 5px'})
                ],
                style={"margin": "60px 0px 30px 0px"},
            ),
            html.Br(),
            # ----------------------------------------------------------------
            # Main section of the interactive visualization
            html.Div(id="figure_panel"),
            # ----------------------------------------------------------------
            # cached data container, local on the browser
            dcc.Store(id="cache_dump", data=str(json.dumps(tmpFileNames))),
        ],
        style={
            "min-height": "100%",
            "width": "900px",
            "text-align": "center",
            "margin": "auto",
        },
    )


def shutdown():
    """
    Server shutdown function, from visual environment.
    """
    # func = request.environ.get("werkzeug.server.shutdown")
    # if func is None:
    #     raise RuntimeError("Not running with the Werkzeug Server")
    # func()
    os.kill(os.getpid(), signal.SIGINT)


def Cluster_Scatter_Plot(object_path):
    """
    Simple scatter plot creation function. This function is used on the
    initial visualization of the data and can be accessed throughout the
    execution of the interactive visualization server.

    Parameters
    ----------
    object_path : str
        The location of the temporary file containing the pickel dump of the
        object we want to visualize.

    """

    # get the necessary data for the visulization
    data_matrix, _, _, number_of_nodes = data_preparation(object_path, 0)

    # create scatter plot with the splitpoint shape
    category_order = {
        "cluster": [
            str(i) for i in range(len(np.unique(data_matrix["cluster"])))
        ]
    }
    color_map = matplotlib.cm.get_cmap("tab20", number_of_nodes)
    colList = {
        str(i): convert_to_hex(color_map(i)) for i in range(color_map.N)
    }

    # create scatter plot
    figure = px.scatter(
        data_matrix,
        x="PC1",
        y="PC2",
        color="cluster",
        category_orders=category_order,
        color_discrete_map=colList,
    )

    # reform visualization
    figure.update_layout(width=850, height=650, plot_bgcolor="#fff")
    figure.update_traces(
        mode="markers",
        marker=dict(size=4),
        hovertemplate=None,
        hoverinfo="skip"
    )
    figure.update_xaxes(
        fixedrange=True,
        showgrid=True,
        gridwidth=1,
        gridcolor="#aaa",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="#aaa",
    )
    figure.update_yaxes(
        fixedrange=True,
        showgrid=True,
        gridwidth=1,
        gridcolor="#aaa",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="#aaa",
    )

    # Markdown description of the figure
    description = dcc.Markdown(
        message_center("des:main_cluser", object_path),
        style={
            "text-align": "left",
            "margin": "-20px 0px 0px 0px",
        }
    )

    return html.Div(
        [
            description,
            dcc.Graph(figure=figure, config={"displayModeBar": False})
        ]
    )


def int_make_scatter_n_hist(
        data_matrix,
        splitPoint,
        bandwidth_scale,
        category_order,
        colList
):
    """
    Create two plots that visualize the data on the second plot and on the
    first give their density representation on the first principal component.

    Parameters
    ----------
    data_matrix : pandas.core.frame.DataFrame
        The projection of the data on the first two Principal Components as
        columns "PC1" and "PC2" and the final cluster each sample belong at
        the end of the algorithm's execution as column "cluster".
    splitPoint : int
        The values of the point the data are split for this plot.
    bandwidth_scale
        Standard deviation scaler for the density aproximation. Allowed values
        are in the (0,1).
    category_order : dict
        The order of witch to show the clusters, contained in the
        visualization, on the legend of the plot.
    colList : dict
        A dictionary containing the color of each cluster (key) as RGBa tuple
        (value).

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        The reulted figure of the function.

    """

    bandwidth = sm.nonparametric.bandwidths.select_bandwidth(
        data_matrix["PC1"],
        "silverman",
        kernel=None
    )
    s, e = FFTKDE(
        kernel="gaussian",
        bw=(bandwidth_scale * bandwidth)
    ).fit(data_matrix["PC1"].to_numpy()).evaluate()

    fig = subplots.make_subplots(
        rows=2, cols=1,
        row_heights=[0.15, 0.85],
        vertical_spacing=0.02,
        shared_yaxes=False,
        shared_xaxes=True,
    )

    fig.add_trace(go.Scatter(
        x=s, y=e,
        mode="lines",
        line=dict(color='royalblue', width=1),
        name='PC1',
        hovertemplate=None,
        hoverinfo="skip",
        showlegend=False
    ), row=1, col=1)
    fig.add_shape(
        type="line",
        yref="y", xref="x",
        xsizemode="scaled",
        ysizemode="scaled",
        x0=splitPoint,
        x1=splitPoint,
        y0=-0.005,
        y1=e.max()*1.2,
        line=dict(color="red", width=1.5),
        row=1, col=1,
    )

    main_figure = px.scatter(
        data_matrix,
        x="PC1", y="PC2",
        color="cluster",
        hover_name="cluster",
        category_orders=category_order,
        color_discrete_map=colList
    )["data"]
    for i in main_figure:
        fig.add_trace(i, row=2, col=1)
    fig.update_traces(
        mode="markers",
        marker=dict(size=4),
        hovertemplate=None,
        hoverinfo="skip",
        row=2, col=1,
    )
    fig.add_shape(
        type="line",
        yref="y", xref="x",
        xsizemode="scaled",
        ysizemode="scaled",
        x0=splitPoint,
        x1=splitPoint,
        y0=data_matrix["PC2"].min() * 1.2,
        y1=data_matrix["PC2"].max() * 1.2,
        line=dict(color="red", width=1.5),
        row=2, col=1,
    )

    # reform visualization
    fig.update_layout(
        width=850, height=700,
        plot_bgcolor="#fff",
        margin={"t": 20, "b": 50},
    )
    fig.update_xaxes(
        fixedrange=True,
        showgrid=True,
        gridwidth=1,
        gridcolor="#aaa",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="#aaa",
    )
    fig.update_yaxes(
        fixedrange=True,
        showgrid=True,
        gridwidth=1,
        gridcolor="#aaa",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="#aaa",
    )

    return fig


def int_make_scatter_n_marginal_scatter(
        data_matrix,
        splitPoint,
        category_order,
        colList,
        centers=None
):
    """
    Create two plots that visualize the data on the second plot and on the
    first give their presentation on the first principal component.

    Parameters
    ----------
    data_matrix : pandas.core.frame.DataFrame
        The projection of the data on the first two Principal Components as
        columns "PC1" and "PC2" and the final cluster each sample belong at
        the end of the algorithm's execution as column "cluster".
    splitPoint : int
        The values of the point the data are split for this plot.
    bandwidth_scale
        Standard deviation scaler for the density aproximation. Allowed values
        are in the (0,1).
    category_order : dict
        The order of witch to show the clusters, contained in the
        visualization, on the legend of the plot.
    colList : dict
        A dictionary containing the color of each cluster (key) as RGBa tuple
        (value).
    centers : numpy.ndarray
        The values of the k-means' centers for the clustering of the data
        projected on the first principal component for each split, for the
        kM-PDDP algorithm.

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        The reulted figure of the function.

    """

    fig = subplots.make_subplots(
        rows=2, cols=1,
        row_heights=[0.15, 0.85],
        vertical_spacing=0.02,
        shared_yaxes=False,
        shared_xaxes=True,
    )

    # Create the marginal scatter plot of the figure (projection on one
    # principal component)
    marginal_figure = px.scatter(
        data_matrix,
        x="PC1",
        y=np.zeros(data_matrix["PC1"].shape[0]),
        color="cluster",
        hover_name="cluster",
        category_orders=category_order,
        color_discrete_map=colList,
    )["data"]
    for i in marginal_figure:
        fig.add_trace(i, row=1, col=1)
    fig.update_traces(
        mode="markers",
        marker=dict(size=5),
        hovertemplate=None,
        hoverinfo="skip",
        showlegend=False,
        row=1, col=1,
    )
    # If there are k-Means centers add them on the marginal scatter plot
    if centers is not None:
        fig.add_trace(go.Scatter(
            x=centers,
            y=np.zeros(2),
            mode="markers",
            marker=dict(symbol=22, color='darkblue', size=15),
            name='centers',
            hovertemplate=None,
            hoverinfo="skip",
            showlegend=False,
        ), row=1, col=1)
    fig.add_shape(
        type="line",
        yref="y", xref="x",
        xsizemode="scaled",
        ysizemode="scaled",
        x0=splitPoint,
        x1=splitPoint,
        y0=-2.5, y1=2.5,
        line=dict(color="red", width=1.5),
        row=1, col=1,
    )

    # Create the main scatter plot of the figure
    main_figure = px.scatter(
        data_matrix,
        x="PC1", y="PC2",
        color="cluster",
        hover_name="cluster",
        category_orders=category_order,
        color_discrete_map=colList,
    )["data"]
    for i in main_figure:
        fig.add_trace(i, row=2, col=1)
    fig.update_traces(
        mode="markers", marker=dict(size=4),
        hovertemplate=None, hoverinfo="skip",
        row=2, col=1,
    )
    fig.add_shape(
        type="line",
        yref="y", xref="x",
        xsizemode="scaled",
        ysizemode="scaled",
        x0=splitPoint,
        x1=splitPoint,
        y0=data_matrix["PC2"].min() * 1.2,
        y1=data_matrix["PC2"].max() * 1.2,
        line=dict(color="red", width=1.5),
        row=2, col=1,
    )

    # Reform visualization
    fig.update_layout(
        width=850,
        height=700,
        plot_bgcolor="#fff",
        margin={"t": 20, "b": 50}
    )
    fig.update_xaxes(
        fixedrange=True, showgrid=True,
        gridwidth=1, gridcolor="#aaa",
        zeroline=True, zerolinewidth=1,
        zerolinecolor="#aaa"
    )
    fig.update_yaxes(
        fixedrange=True, showgrid=True,
        gridwidth=1, gridcolor="#aaa",
        zeroline=True, zerolinewidth=1,
        zerolinecolor="#aaa"
    )

    return fig


def recalculate_after_spchange(hipart_object, split, splitpoint_value):
    """
    Given the serial number of the HiPart algorithm tree`s internal nodes and a
    new splitpoint value recreate the results of the HiPart member algorithm
    with the indicated change.

    Parameters
    ----------
    hipart_object : dePDDP or iPDDP or kM_PDDP or PDDP object
        The object that we want to manipulate on the premiss of this function.
    split : int
        The serial number of the dePDDP tree`s internal nodes.
    splitpoint_value : float
        New splitpoint value.

    Returns
    -------
    hipart_object : dePDDP or iPDDP or kM_PDDP or PDDP object
        A dePDDP class type object, with complete results on the algorithm's
        analysis

    """

    tree = hipart_object.tree

    # find the cluster nodes
    clusters = tree.leaves()
    clusters = sorted(clusters, key=lambda x: x.identifier)

    # find the tree`s internal nodes a.k.a. splits
    dictionary_of_nodes = tree.nodes
    number_of_nodes = len(list(dictionary_of_nodes.keys()))
    leaf_node_list = [j.identifier for j in clusters]
    internal_nodes = [
        i for i in range(number_of_nodes) if not (i in leaf_node_list)
    ]

    # Insure that the root node will be always an internal node while is the
    # only node in the tree
    if len(internal_nodes) == 0:
        internal_nodes = [0]

    # remove all the splits (nodes) of the tree after the one bing manipulated.
    # this process starts from the last to the first to ensure deletion
    node_keys = list(dictionary_of_nodes.keys())
    for i in range(len(node_keys) - 1, 0, -1):
        if node_keys[i] > internal_nodes[split]:
            if tree.get_node(node_keys[i]) is not None:
                if not tree.get_node(internal_nodes[split]).is_root():
                    if (
                        tree.parent(internal_nodes[split]).identifier
                        != tree.parent(node_keys[i]).identifier
                    ):
                        tree.remove_node(node_keys[i])
                else:
                    tree.remove_node(node_keys[i])

    # change the split permition of all the internal nodes to True so the
    # algorithm can execute correctly
    dictionary_of_nodes = tree.nodes
    for i in dictionary_of_nodes:
        if dictionary_of_nodes[i].is_leaf():
            if dictionary_of_nodes[i].data["split_criterion"] is not None:
                dictionary_of_nodes[i].data["split_permition"] = True

    # reset status variables for the code to execute
    hipart_object.node_ids = len(list(dictionary_of_nodes.keys())) - 1
    hipart_object.cluster_color = len(tree.leaves()) + 1
    tree.get_node(internal_nodes[split]).data["splitpoint"] = splitpoint_value

    # continue the algorithm`s execution from the point left
    hipart_object.tree = tree
    hipart_object.tree = partial_predict(hipart_object)

    return hipart_object


def partial_predict(hipart_object):
    """
    Execute the steps of the algorithm dePDDP untill one of the two stopping
    creterion is not true.

    Parameters
    ----------
    hipart_object : dePDDP or iPDDP or kM_PDDP or PDDP object
        The object that we want to manipulated on the premiss of this function.

    Returns
    -------
    tree : treelib.Tree
        The newlly created tree after the execution of the algorithm.

    """

    tree = hipart_object.tree

    found_clusters = len(tree.leaves())
    selected_node = hipart_object.select_kid(tree.leaves())

    while (
        (selected_node is not None)
        and (found_clusters < hipart_object.max_clusters_number)
    ):  # (ST1) or (ST2)

        hipart_object.split_function(tree, selected_node)  # step (1)

        # select the next kid for split based on the local minimum density
        selected_node = hipart_object.select_kid(tree.leaves())  # step (2)
        found_clusters = found_clusters + 1  # (ST1)

    return tree
