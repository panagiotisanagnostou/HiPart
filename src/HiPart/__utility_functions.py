# Copyright (c) 2022 Panagiotis Anagnostou

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Utility functions of the HiPart package.

@author: Panagiotis Anagnostou
@author: Nicos Pavlidis
"""

import copy
import math
import matplotlib
import numpy as np
import statsmodels.api as sm
import warnings

from KDEpy import FFTKDE
from scipy import stats as st
from scipy.optimize import Bounds, minimize, NonlinearConstraint, SR1
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA, FastICA


def band_const(d):
    """
    Calculate the data bandwidth based in their dimensions.

    Parameters
    ----------
    d : int
        Number of dimensions

    Returns
    -------
    float
        Bandwidth
    """
    return 0.45 / np.power(d, 0.2)


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
    two_dimentions : bool
        If True the projection will be on the first two components of the 'pca'
        and 'ica' methods. The 'kpca' and 'tsne' methods will be projected only
        on one dimension because of the nature.
    decomposition_args : dict
        Arguments to use by each of the decomposition methods utilized by the
        HiPart package.

    Returns
    -------
    two_dimensions : numpy.ndarray
        The projections of the samples on the first two components of the pca
        and kernel pca methods.

    """
    if two_dimentions:
        if decomposition_method in ["tsne"]:
            raise ValueError(
                ": The decomposition method ("
                + decomposition_method
                + ") cannot be executed correctly for two dimentions!"
            )
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
    elif decomposition_method == "tsne":
        tsne = TSNE(
            n_components=n_of_dimentions,
            **decomposition_args
        )
        two_dimensions = tsne.fit_transform(data_matrix)
    else:
        raise ValueError(
            ": The decomposition method ("
            + decomposition_method
            + ") is not supported!"
        )

    return two_dimensions


def initialize_b(x0, X, depth_init=True):
    """
    Initialize the b parameter of the MDH algorithm. The initialization is done
    by finding the minimum density point of the projection of the data matrix
    onto the vector v.

    Parameters
    ----------
    x0 : numpy.ndarray (1D)
        The vector v in the first d position of x0 vector and the point b in the
        last one. The vector v is the projection direction of the data matrix
        and b is a random estimation of the minimum density point.
    X : numpy.ndarray (2D)
        The data matrix.
    depth_init : bool (default=True)
        If (True) the b parameter is initialized by finding the maximum depth of
        the curve created by the projection of the data matrix onto the vector
        v. If (False) the b parameter is initialized by finding the lowest local
        minimum density point of the projection of the data matrix onto the
        vector v.

    Returns
    -------
    b : float
        The initialized b parameter on the local minimum density point of the
        projection of the data matrix onto the vector v.

    """
    v = x0[:-1] / np.linalg.norm(x0[:-1])
    xv = np.sort(np.dot(X, v))
    # Bandwidth calculation
    h = np.std(xv) * band_const(X.shape[0])

    # The unitization of vector v produces a vector field with nans due to the norm
    try:
        x_ticks, y = FFTKDE(kernel="gaussian", bw=h).fit(xv).evaluate()
    except ValueError:
        return np.array([])

    # find all local maxima of the projection's density
    maxima = np.where(np.diff(1 * (np.diff(y) > 0)) == -1)[0]

    # RuntimeErrors are raised if there are no local maxima this is not possibl
    # by the definition of the KDE.
    if len(maxima) == 0:
        raise RuntimeError("MDH: no local maximum: This shouldn't be possible!")
    # If there is only one maximum, then the distribution is uni-modal and there
    # is no need for further searching for a local minimum.
    elif len(maxima) == 1:
        warnings.warn(
            "MDH: uni-modal distribution there is no need for further processing!")
        return np.array([])

    # locations of maxima in original y array
    # maxima += 1  # #$$# den exei shmasia an einai maxima h maxima+1 kanonika
    # #$$# tha prepei an einai sthn mesh outos h allos

    if depth_init:
        # find minima between every pair of peaks
        depth = np.inf
        p = None
        for i in np.arange(len(maxima) - 1):
            pos = maxima[i] + np.argmin(y[maxima[i]:maxima[i + 1]])
            # inverse of depth (to avoid numerical difficulties)
            tmp_depth = np.amin(
                [np.amax(y[maxima[:i + 1]]), np.amax(y[maxima[i + 1:]])]) - y[
                            pos]
            # No divisions by zero
            if tmp_depth:
                d = y[pos] / tmp_depth
            else:
                continue

            # Find the minimum depth
            if d < depth:
                depth = d
                p = pos

        # Return, if exists, the location of the minimum density point of the projection
        if p:
            return x_ticks[p]
        else:
            return None

    else:
        # Find the lowest minimum between any peaks
        pos = maxima[0] + np.argmin(y[maxima[0]:maxima[-1] + 1])
        return x_ticks[pos]


def md_sqp(x0, X, k):
    """
    MDH through SQP in Python

    Parameters
    ----------
    x0 : numpy.ndarray, shape (d,)
        Initial point (v,b) for optimisation algorithm .The vector v is the
        projection direction of the data matrix and b is a random estimation of
        the minimum density point.
    X : ndarray, shape (n,d)
        N times D Input data matrix. Must be centred
    k : float
        Range parameter

    Returns
    -------
    res : OptimizeResult
        Output from the minimizer.
    depth : float
        The depth of the minimum density point of the projection of the data.

    """

    # I need this to avoid division by zero in the Jacobian
    # kern = np.sqrt(np.finfo("float").eps)

    ub = np.ones(len(x0))
    lb = np.full(len(x0), -1, dtype=float)
    lb[0] = 0

    if k == 0:
        # pad last entry with a zero
        def Fx(x):
            return fx(np.append(x, 0), X)

        # remove last derivative
        def JacF(x):
            return (dKdeDvb(np.append(x, 0), X))[:-1]

        def cons_f(x):
            return np.sum(x ** 2)

        def cons_J(x):
            return 2 * x

        def cons_H(x, v):
            return v[0] * 2 * np.eye(len(x))
    else:
        def Fx(x):
            return fx(x, X)

        def JacF(x):
            return dKdeDvb(x, X)

        # l2_norm = 1 constraint
        def cons_f(x):
            return np.sum(x[:-1] ** 2)

        def cons_J(x):
            return np.append(2 * x[:-1], 0)

        def cons_H(x, v):
            return v[0] * 2 * np.diag(
                np.append(np.ones(len(x) - 1), 0))

        ub[-1] = k
        lb[-1] = -k

    nlc = NonlinearConstraint(cons_f, 1, 1, jac=cons_J, hess=cons_H)

    x0 = x0 if x0[0] > 0 else -x0
    res = minimize(
        Fx, x0,
        method="trust-constr",
        jac=JacF,
        hess=SR1(),
        constraints=nlc,
        options={"verbose": 0},
        bounds=Bounds(lb, ub),
    )

    if res:
        return res, Fx(res.x)
    else:
        return res, None


def fx(xcur, X):
    """
    Evaluate the 1D KDE at xcur[-1] after projecting the data matrix X onto the
    unit-vector xcur[:-1].

    Parameters
    ----------
    xcur: numpy.ndarray, shape (d,)
        Point (v,b). The vector v is the projection direction of the data matrix
        X and b is a random estimation of the minimum density point.
    X: numpy.ndarray, shape (n,d)
        N times D Input data matrix.

    Returns
    -------
    float
        The value of the 1D KDE at b after projecting the data matrix X onto the
        unit-vector v.

    """
    # evaluate 1D kde at xcur[-1] after projecting onto unit-vector xcur[:-1]
    v = xcur[:-1]
    b = xcur[-1]

    # compute projections
    xv = np.dot(X, v)
    band = np.std(xv) * band_const(len(xv))

    return np.mean(st.norm.pdf(xv, b, band))


def dKdeDvb(xcur, X):
    """
    Evaluate the derivative of the 1D KDE at v after projecting the data
    matrix X onto the unit-vector b.

    Parameters
    ----------
    xcur : numpy.ndarray, shape (d,)
        Point (v,b). The vector v (xcur[:-1]) is the projection direction of the
        data matrix X and b (xcur[-1]) is a random estimation of the minimum
        density point.
    X : numpy.ndarray, shape (n,d)
        N times D Input data matrix.

    Returns
    -------
    numpy.ndarray, shape (d,)
        The derivative of the 1D KDE at v after projecting the data matrix X.

    """

    v = xcur[:-1]
    b = xcur[-1]

    # compute projection of X onto v
    proj = np.dot(X, v)
    N = proj.shape[0]
    bn = band_const(N)

    # Bandwidth calculation
    h = np.std(proj) * bn

    # ==========================================================================
    # Derivative of fx w.r.t. projected points and bandwidth
    DgF = np.empty(N + 1, dtype=float)
    # first N entries are derivatives w.r.t. each projected point
    kde = st.norm.pdf(proj, b, h)
    DgF[:-1] = kde * (b - proj) / (h * h * N)
    # derivative w.r.t. bandwidth
    DgF[-1] = -np.mean(kde) / h + np.mean(kde * (np.power((b - proj), 2))) / (
            h * h * h)

    # ==========================================================================
    # Derivative of g = (p_1,p_2, ... p_N,h) w.r.t. full-dimensional projection vector!
    # Data is CENTRED and X stores observations in rows: Thoroughly debugged
    # get std of projected data
    dhdv = (bn ** 2) / (h * (N - 1))

    # last parenthesis: De-means all columns of X
    last_row = dhdv * (proj @ X)
    DvG = np.vstack((X, last_row))

    # ==========================================================================
    # derivative of fx w.r.t to xcur = (v,b)
    out = np.empty(len(xcur), dtype=float)
    out[:-1] = DgF @ DvG
    out[-1] = np.sum((proj - b) * kde) / (h * h * len(proj))

    return out


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
            need to pre-scale your features."""
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
    bandwidth_scale
    scaler
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
        hist.plot(s, e * (PP.shape[0] / scaler))
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
        A list of lists containing all the paths to the leaves of the tree.
    i : int
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
    Create the linkage matrix for the encoding of the divisive clustering tree
    created by the member algorithms of the HiPart package.

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
        # the samples of the node on same level until only one node remains
        if tree.get_node(i).is_leaf():
            if not tree.get_node(i).data["dendrogram_check"]:
                # Set all the samples of the included in the node/cluster as
                # unlinked nodes on the dendrogram tree
                tree.get_node(i).data["unlinked_nodes"] = tree.get_node(
                    i
                ).data["indices"]
                # Create the dendrogram`s subtree and update the algorithm tree
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
                        children[-1].data["counts"] + children[-2].data[
                            "counts"],
                    ]]
                )
                # Update of the data of the algorithm execution tree node
                tree.get_node(
                    i
                ).data["dendromgram_indicator"] = dendrogram_counts
                tree.get_node(i).data["counts"] = (
                        children[-1].data["counts"]
                        + children[-2].data["counts"]
                )
                tree.get_node(i).data["unlinked_nodes"] = [dendrogram_counts]
                dendrogram_counts += 1

    # Remove the first row of the linkage matrix because it is the
    # initialization`s row of zeros
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
        The next free node of the dendrogram subtree.
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

        # Add the last unlinked node`s subtree
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
