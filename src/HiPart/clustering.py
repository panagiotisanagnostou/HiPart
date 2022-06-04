# -*- coding: utf-8 -*-
"""
Implementation of the clustering algorithms, members of the HiPart package.
"""

import HiPart.__utility_functions as util
import numpy as np
import statsmodels.api as sm

from KDEpy import FFTKDE
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans
from treelib import Tree


class dePDDP:
    """
    Class dePDDP. It executes the dePDDP algorithm.

    References
    ----------
    Tasoulis, S. K., Tasoulis, D. K., & Plagianakos, V. P. (2010). Enhancing
    principal direction divisive clustering. Pattern Recognition, 43(10), 3391-
    3411.

    Parameters
    ----------
    decomposition_method : str, optional
        One of the ('pca', 'kpca', 'ica') supported decomposition methods used
        as kernel for the dePDDP algorithm.
    max_clusters_number : int, optional
        Desired maximum number of clusters to find the dePDDP algorithm.
    split_data_bandwidth_scale : float, optional
        Standard deviation scaler for the density aproximation.
    percentile : float, optional
        The peprcentile distance from the dataset's edge in which a split can
        not occur. [0,0.5) values are allowed.
    min_sample_split : int, optional
        The minimum number of points needed in a cluster for a split to occur.
    visualization_utility : bool, optional
        If (True) generate the data needed by the visualization utilities of
        the package othrerwise, if false the split_visualization and
        interactive_visualization of the package can not be created.
    **decomposition_args :
        Arguments for each of the decomposition methods ("PCA" as "pca",
        "KernelPCA" as "kpca", "FastICA" as "ica") utilized by the HiPart
        package, as documented in the scikit-learn package, from which they are
        implemented.

    Attributes
    ----------
    output_matrix : numpy.ndarray
        Model's step by step execution output.
    labels_ :
        Extracted clusters from the algorithm

    """

    def __init__(
        self,
        decomposition_method="pca",
        max_clusters_number=100,
        bandwidth_scale=0.5,
        percentile=0.1,
        min_sample_split=5,
        visualization_utility=True,
        **decomposition_args
    ):
        self.decomposition_method = decomposition_method
        self.max_clusters_number = max_clusters_number
        self.split_data_bandwidth_scale = bandwidth_scale
        self.percentile = percentile
        self.min_sample_split = min_sample_split
        self.visualization_utility = visualization_utility
        self.decomposition_args = decomposition_args

    def fit(self, X):
        """
        Execute the dePDDP algorithm and return all the execution data in the
        form of a dePDDP class object.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix with the samples on the rows and the avriables on the
            columns.

        Returns
        -------
        self
            A dePDDP class type object, with complete results on the
            algorithm's analysis.

        """
        self.X = X
        self.samples_number = X.shape[0]

        # create an id vector for the samples of X
        indices = np.array([int(i) for i in range(np.size(self.X, 0))])

        # initialize tree and root node                 # step (0)
        tree = Tree()
        # nodes unique IDs indicator
        self.node_ids = 0
        # nodes next color indicator (necessary for visualization purposes)
        self.cluster_color = 0
        tree.create_node(
            tag="cl_" + str(self.node_ids),
            identifier=self.node_ids,
            data=self.calculate_node_data(indices, self.X, self.cluster_color),
        )
        # inidcator for the next node to split
        selected_node = 0

        # if no possibility of split exists on the data     # (ST2)
        if not tree.get_node(0).data["split_permition"]:
            raise RuntimeError("cannot split the data at all!!!")

        # Initialize the ST1 stopping critirion counter that count the number
        # of clusters                                       # (ST1)
        found_clusters = 1
        while (
            (selected_node is not None)
            and (found_clusters < self.max_clusters_number)
        ):  # (ST1) or (ST2)

            self.split_function(tree, selected_node)  # step (1)

            # select the next kid for split based on the local minimum density
            selected_node = self.select_kid(tree.leaves())  # step (2)
            found_clusters = found_clusters + 1  # (ST1)

        self.tree = tree
        return self

    def fit_predict(self, X):
        """
        Execute the dePDDP algorithm and return the results of the execution
        in the form of labels.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix with the samples on the rows and the avriables on the
            columns.

        Returns
        -------
        labels_ : numpy.ndarray
            Extracted clusters from the algorithm.

        """

        return self.fit(X).labels_

    def split_function(self, tree, selected_node):
        """
        Split the indicated node on the minimum of the local minimum density
        of the data projected on the first principal component.

        Because python passes by refference data this function doesn't need a
        return statment.

        Parameters
        ----------
        tree : treelib.tree.Tree
            The tree build by the dePDDP algorithm, in order to cluster the
            input data.

        Returns
        -------
            There no returns in this function. The results of this funciton
            pass to execution by utilizing the python's pass-by-reference
            nature.

        """
        node = tree.get_node(selected_node)
        node.data["split_permition"] = False

        # left child indecies extracted from the nodes splitpoint and the
        # indecies included in the parent node
        left_kid_index = node.data["indices"][
            np.where(
                node.data["projection"][:, 0] < node.data["splitpoint"]
            )[0]
        ]
        # right child indecies
        right_kid_index = node.data["indices"][
            np.where(
                node.data["projection"][:, 0] >= node.data["splitpoint"]
            )[0]
        ]

        # Nodes and data creation for the children
        # Uses the calculate_node_data function to create the data for the node
        tree.create_node(
            tag="cl" + str(self.node_ids + 1),
            identifier=self.node_ids + 1,
            parent=node.identifier,
            data=self.calculate_node_data(
                left_kid_index, self.X[left_kid_index, :],
                node.data["color_key"]
            ),
        )
        tree.create_node(
            tag="cl" + str(self.node_ids + 2),
            identifier=self.node_ids + 2,
            parent=node.identifier,
            data=self.calculate_node_data(
                right_kid_index, self.X[right_kid_index, :],
                self.cluster_color + 1,
            ),
        )

        self.cluster_color += 1
        self.node_ids += 2

    def select_kid(self, leaves):
        """
        The clusters each time exist in the leaves of the trees. From those
        leaves select the next leave to split based on the algorithm's
        specifications.

        This function creates the nescesary cause for the stopping criterion
        ST1.

        Parameters
        ----------
        leaves : list of treelib.node.Node
            The list of nodes needed to exam to select the next Node to split.

        Returns
        -------
        next_split : int
            The identifier of the next node to split by the algorithm.

        """
        next_split = None

        # Remove the nodes that can not split further
        leaves = list(
            np.array(leaves)[
                [
                    True if i.data["split_criterion"] is not None else False
                    for i in leaves
                ]
            ]
        )

        if len(leaves) > 0:
            for i in sorted(
                enumerate(leaves), key=lambda x: x[1].data["split_criterion"]
            ):
                if i[1].data["split_permition"]:
                    next_split = i[1].identifier
                    break

        return next_split

    def calculate_node_data(self, indices, data_matrix, key):
        """
        Calculation of the projections onto the Principal Components with the
        utilization of the "Principal Components Analysis" or the "Kernel
        Principal Components Analysis" or the "Indipented Component Analysis"
        methods.

        Determination of the projection's density and search for its local
        minima. The lowest local minimum point within the allowed sample
        percentiles of the projection's density representation is selected
        as the split point.

        This function leads to the second Stopping criterion 2 of the
        algorithm.

        Parameters
        ----------
        indices : numpy.ndarray
            The index of the samples in the original data matrix.
        data_matrix : numpy.ndarray
            The data matrix containing all the data for the samples.
        key : int
            The value of the color for each node.

        Returns
        -------
        data : dict
            The necesary data for each node which are spliting point.

        """
        # if the number of samples
        if indices.shape[0] > self.min_sample_split:
            # execute pca on the data matrix
            projection = util.execute_decomposition_method(
                data_matrix=data_matrix,
                decomposition_method=self.decomposition_method,
                two_dimentions=self.visualization_utility,
                decomposition_args=self.decomposition_args,
            )
            one_dimension = projection[:, 0]

            # calculate the standared deviation of the data
            bandwidth = sm.nonparametric.bandwidths.select_bandwidth(
                one_dimension, "silverman", kernel=None
            )

            # calculate the density function on the 1st Princpal Component
            # x_ticks: projection points on the 1st PC
            # evaluation: the density of the projections on the 1st PC
            x_ticks, evaluation = (
                FFTKDE(
                    kernel="gaussian",
                    bw=self.split_data_bandwidth_scale * bandwidth
                )
                .fit(one_dimension)
                .evaluate()
            )
            # calculate all the local minima
            minimum_indices = argrelextrema(evaluation, np.less)[0]

            # Find the location of the local minima and make sure they are
            # with in the given percentile limits
            quantile_value = np.quantile(
                one_dimension,
                (self.percentile, (1 - self.percentile))
            )
            local_minimum_index = np.where(
                np.logical_and(
                    x_ticks[minimum_indices] > quantile_value[0],
                    x_ticks[minimum_indices] < quantile_value[1],
                )
            )

            # List all the numbers for the local minima (ee) and their
            # respective position (ss) on the 1st PC.
            ss = x_ticks[minimum_indices][local_minimum_index]
            ee = evaluation[minimum_indices][local_minimum_index]

            # if there is at least one local minima split the data
            if ss.size > 0:
                minimum_location = np.argmin(ee)
                splitpoint = ss[minimum_location]
                split_criterion = ee[minimum_location]
                flag = True
            else:
                splitpoint = None  # (ST2)
                split_criterion = None  # (ST2)
                flag = False  # (ST2)
        # =========================
        else:
            projection = None  # (ST2)
            splitpoint = None  # (ST2)
            split_criterion = None  # (ST2)
            flag = False  # (ST2)

        return {
            "indices": indices,
            "projection": projection,
            "splitpoint": splitpoint,
            "split_criterion": split_criterion,
            "split_permition": flag,
            "color_key": key,
            "dendrogram_check": False,
        }

    @property
    def decomposition_method(self):
        return self._decomposition_method

    @decomposition_method.setter
    def decomposition_method(self, v):
        if not (v in ["pca", "kpca", "ica"]):
            raise ValueError(
                "decomposition_method: "
                + str(v)
                + ": Unknown decomposition method!"
            )
        self._decomposition_method = v

    @property
    def max_clusters_number(self):
        return self._max_clusters_number

    @max_clusters_number.setter
    def max_clusters_number(self, v):
        if v < 0 or (not isinstance(v, int)):
            raise ValueError("min_sample_split: "
                             + "Invalid value it should be int and > 1")
        self._max_clusters_number = v

    @property
    def split_data_bandwidth_scale(self):
        return self._split_data_bandwidth_scale

    @split_data_bandwidth_scale.setter
    def split_data_bandwidth_scale(self, v):
        if v <= 0:
            raise ValueError(
                "split_data_bandwidth_scale: Should be > 0"
            )
        self._split_data_bandwidth_scale = v

    @property
    def percentile(self):
        return self._percentile

    @percentile.setter
    def percentile(self, v):
        if v >= 0.5 or v < 0:
            raise ValueError("percentile: Should be between [0,0.5) interval")
        self._percentile = v

    @property
    def min_sample_split(self):
        return self._min_sample_split

    @min_sample_split.setter
    def min_sample_split(self, v):
        if v < 0 or (not isinstance(v, int)):
            raise ValueError("min_sample_split: "
                             + "Invalid value it should be int and > 1")
        self._min_sample_split = v

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, v):
        self._tree = v

    @property
    def output_matrix(self):
        ndDict = self.tree.nodes
        output_matrix = [np.zeros(np.size(self.X, 0))]
        for i in ndDict:
            if not ndDict[i].is_leaf():
                # create output cluster spliting matrix
                tmp = np.copy(output_matrix[-1])
                tmp[
                    self.tree.children(i)[0].data["indices"]
                ] = self.tree.children(i)[0].identifier
                tmp[
                    self.tree.children(i)[1].data["indices"]
                ] = self.tree.children(i)[1].identifier
                output_matrix.append(tmp)
        del output_matrix[0]
        output_matrix = np.array(output_matrix).transpose()
        self.output_matrix = output_matrix
        return self._output_matrix

    @output_matrix.setter
    def output_matrix(self, v):
        self._output_matrix = v

    @property
    def labels_(self):
        labels_ = np.ones(np.size(self.X, 0))
        for i in self.tree.leaves():
            labels_[i.data["indices"]] = i.identifier
        self.labels_ = labels_
        return self._labels_

    @labels_.setter
    def labels_(self, v):
        self._labels_ = v


class iPDDP:
    """
    Class iPDDP. It executes the iPDDP algorithm.

    References
    ----------
    Tasoulis, S. K., Tasoulis, D. K., & Plagianakos, V. P. (2010). Enhancing
    principal direction divisive clustering. Pattern Recognition, 43(10), 3391-
    3411.

    Parameters
    ----------
    decomposition_method : str, optional
        One of the ('pca', 'kpca', 'ica') supported decomposition methods used
        as kernel for the iPDDP algorithm.
    max_clusters_number : int, optional
        Desired maximum number of clusters for the algorithm.
    percentile : float, optional
        The peprcentile distance from the dataset's edge in which a split can
        not occur. [0,0.5) values are allowed.
    min_sample_split : int, optional
        The minimum number of points needed in a cluster for a split to occur.
    visualization_utility : bool, optional
        If (True) generate the data needed by the visualization utilities of
        the package othrerwise, if false the split_visualization and
        interactive_visualization of the package can not be created.
    **decomposition_args :
        Arguments for each of the decomposition methods ("PCA" as "pca",
        "KernelPCA" as "kpca", "FastICA" as "ica") utilized by the HiPart
        package, as documented in the scikit-learn package, from which they are
        implemented.

    Attributes
    ----------
    output_matrix : numpy.ndarray
        Model's step by step execution output.
    labels_ :
        Extracted clusters from the algorithm.

    """

    def __init__(
        self,
        decomposition_method="pca",
        max_clusters_number=100,
        percentile=0.1,
        min_sample_split=5,
        visualization_utility=True,
        **decomposition_args
    ):
        self.decomposition_method = decomposition_method
        self.max_clusters_number = max_clusters_number
        self.percentile = percentile
        self.min_sample_split = min_sample_split
        self.visualization_utility = visualization_utility
        self.decomposition_args = decomposition_args

    def fit(self, X):
        """
        Execute the iPDDP algorithm and return all the execution data in the
        form of a iPDDP class object.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix with the samples on the rows and the avriables on the
            columns.

        Returns
        -------
        self
            A iPDDP class type object, with complete results on the
            algorithm's analysis.

        """
        self.X = X
        self.samples_number = X.shape[0]

        # create an id vector for the samples of X
        indices = np.array([int(i) for i in range(np.size(self.X, 0))])

        # initialize tree and root node                 # step (0)
        tree = Tree()
        # nodes unique IDs indicator
        self.node_ids = 0
        # nodes next color indicator (necessary for visualization purposes)
        self.cluster_color = 0
        tree.create_node(
            tag="cl_" + str(self.node_ids),
            identifier=self.node_ids,
            data=self.calculate_node_data(indices, self.X, self.cluster_color),
        )
        # inidcator for the next node to split
        selected_node = 0

        # if no possibility of split exists on the data     # (ST2)
        if not tree.get_node(0).data["split_permition"]:
            raise RuntimeError("cannot split the data at all!!!")

        # Initialize the ST1 stopping critirion counter that count the number
        # of clusters                                       # (ST1)
        found_clusters = 1
        while (
            (selected_node is not None)
            and (found_clusters < self.max_clusters_number)
        ):  # (ST1) or (ST2)

            self.split_function(tree, selected_node)  # step (1)

            # select the next kid for split based on the local minimum density
            selected_node = self.select_kid(tree.leaves())  # step (2)
            found_clusters = found_clusters + 1  # (ST1)

        self.tree = tree

        return self

    def fit_predict(self, X):
        """
        Execute the iPDDP algorithm and return the results of the execution
        in the form of labels.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix with the samples on the rows and the avriables on the
            columns.

        Returns
        -------
        labels_ : numpy.ndarray
            Extracted clusters from the algorithm.

        """

        return self.fit(X).labels_

    def split_function(self, tree, selected_node):
        """
        Split the indicated node on the maximum gap between two consecutive
        points of the data projected on the first principal component.

        Because python passes by refference data this function doesn't need a
        return statment.

        Parameters
        ----------
        tree : treelib.tree.Tree
            The tree build by the iPDDP algorithm, in order to cluster the
            input data.

        Returns
        -------
            There no returns in this function. The results of this funciton
            pass to execution by utilizing the python's pass-by-reference
            nature.

        """

        node = tree.get_node(selected_node)
        node.data["split_permition"] = False

        # left child indecies extracted from the nodes splitpoint and the
        # indecies included in the parent node
        left_kid_index = node.data["indices"][
            np.where(
                node.data["projection"][:, 0] < node.data["splitpoint"]
            )[0]
        ]
        # right child indecies
        right_kid_index = node.data["indices"][
            np.where(
                node.data["projection"][:, 0] >= node.data["splitpoint"]
            )[0]
        ]

        # Nodes and data creation for the children
        # Uses the calculate_node_data function to create the data for the node
        tree.create_node(
            tag="cl" + str(self.node_ids + 1),
            identifier=self.node_ids + 1,
            parent=node.identifier,
            data=self.calculate_node_data(
                left_kid_index,
                self.X[left_kid_index, :],
                node.data["color_key"],
            ),
        )
        tree.create_node(
            tag="cl" + str(self.node_ids + 2),
            identifier=self.node_ids + 2,
            parent=node.identifier,
            data=self.calculate_node_data(
                right_kid_index,
                self.X[right_kid_index, :],
                self.cluster_color + 1,
            ),
        )

        self.cluster_color += 1
        self.node_ids += 2

    def select_kid(self, leaves):
        """
        The clusters each time exist in the leaves of the trees. From those
        leaves select the next leave to split based on the algorithm's
        specifications.

        This function creates the nescesary cause for the stopping criterion
        ST1.

        Parameters
        ----------
        leaves : list of treelib.node.Node
            The list of nodes needed to exam to select the next Node to split.

        Returns
        -------
        next_split : int
            The identifier of the next node to split by the algorithm.

        """
        next_split = None

        # Remove the nodes that can not split further
        leaves = list(
            np.array(leaves)[
                [
                    True if i.data["split_criterion"] is not None else False
                    for i in leaves
                ]
            ]
        )

        if len(leaves) > 0:
            for i in sorted(
                enumerate(leaves),
                key=lambda x: x[1].data["split_criterion"],
                reverse=True,
            ):
                if i[1].data["split_permition"]:
                    next_split = i[1].identifier
                    break

        return next_split

    def calculate_node_data(self, indices, data_matrix, key):
        """
        Calculation of the projections onto the Principal Components with the
        utilization of the "Principal Components Analysis" or the "Kernel
        Principal Components Analysis" or the "Indipented Component Analysis"
        methods.

        Determination of the projection's maximum distance between to
        consecutive points and choses it as the splitpoint for this node.

        This function leads to the second Stopping criterion 2 of the
        algorithm.

        Parameters
        ----------
        indices : ndarray of shape (n_samples,)
            The index of the samples in the original data matrix.
        data_matrix : ndarray of shape (n_samples, n_dims)
            The data matrix containing all the data for the samples.
        key : int
            The value of the color for each node.

        Returns
        -------
        data : dict
            The necesary data for each node which are spliting point.

        """

        # if the number of samples
        if indices.shape[0] > self.min_sample_split:
            # execute pca on the data matrix
            projection = util.execute_decomposition_method(
                data_matrix=data_matrix,
                decomposition_method=self.decomposition_method,
                two_dimentions=self.visualization_utility,
                decomposition_args=self.decomposition_args,
            )
            one_dimension = projection[:, 0]

            sort_indices = np.argsort(one_dimension)
            projection = projection[sort_indices, :]
            one_dimension = projection[:, 0]
            indices = indices[sort_indices]

            quantile_value = np.quantile(
                one_dimension,
                (self.percentile, (1 - self.percentile))
            )
            within_limits = np.where(
                np.logical_and(
                    one_dimension > quantile_value[0],
                    one_dimension < quantile_value[1],
                )
            )[0]

            # if there is at least one split the allowed percentile of the data
            if within_limits.size > 0:
                distances = np.diff(one_dimension[within_limits])
                loc = np.where(np.isclose(distances, np.max(distances)))[0][0]
                splitpoint = one_dimension[within_limits][loc] + distances[loc] / 2
                split_criterion = distances[loc]
                flag = True
            else:
                splitpoint = None  # (ST2)
                split_criterion = None  # (ST2)
                flag = False  # (ST2)
        # =========================
        else:
            projection = None
            splitpoint = None  # (ST2)
            split_criterion = None  # (ST2)
            flag = False  # (ST2)

        return {
            "indices": indices,
            "projection": projection,
            "splitpoint": splitpoint,
            "split_criterion": split_criterion,
            "split_permition": flag,
            "color_key": key,
            "dendrogram_check": False,
        }

    @property
    def decomposition_method(self):
        return self._decomposition_method

    @decomposition_method.setter
    def decomposition_method(self, v):
        if not (v in ["pca", "kpca", "ica"]):
            raise ValueError(
                "decomposition_method: "
                + str(v)
                + ": Unknown decomposition method!"
            )
        self._decomposition_method = v

    @property
    def max_clusters_number(self):
        return self._max_clusters_number

    @max_clusters_number.setter
    def max_clusters_number(self, v):
        if v < 0 or (not isinstance(v, int)):
            raise ValueError("min_sample_split: "
                             + "Invalid value it should be int and > 1")
        self._max_clusters_number = v

    @property
    def percentile(self):
        return self._percentile

    @percentile.setter
    def percentile(self, v):
        if v >= 0.5 or v < 0:
            raise ValueError("percentile: Should be between [0,0.5) interval")
        self._percentile = v

    @property
    def min_sample_split(self):
        return self._min_sample_split

    @min_sample_split.setter
    def min_sample_split(self, v):
        if v < 0 or (not isinstance(v, int)):
            raise ValueError("min_sample_split: "
                             + "Invalid value it should be int and > 1")
        self._min_sample_split = v

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, v):
        self._tree = v

    @property
    def output_matrix(self):
        ndDict = self.tree.nodes
        output_matrix = [np.zeros(np.size(self.X, 0))]
        for i in ndDict:
            if not ndDict[i].is_leaf():
                # create output cluster spliting matrix
                tmp = np.copy(output_matrix[-1])
                tmp[
                    self.tree.children(i)[0].data["indices"]
                ] = self.tree.children(i)[0].identifier
                tmp[
                    self.tree.children(i)[1].data["indices"]
                ] = self.tree.children(i)[1].identifier
                output_matrix.append(tmp)
        del output_matrix[0]
        output_matrix = np.array(output_matrix).transpose()
        self.output_matrix = output_matrix
        return self._output_matrix

    @output_matrix.setter
    def output_matrix(self, v):
        self._output_matrix = v

    @property
    def labels_(self):
        labels_ = np.ones(np.size(self.X, 0))
        for i in self.tree.leaves():
            labels_[i.data["indices"]] = i.identifier
        self.labels_ = labels_
        return self._labels_

    @labels_.setter
    def labels_(self, v):
        self._labels_ = v


class kM_PDDP:
    """
    Class kM-PDDP. It executes the kM-PDDP algorithm.

    References
    ----------
    Zeimpekis, D., & Gallopoulos, E. (2008). Principal direction divisive
    partitioning with kernels and k-means steering. In Survey of Text Mining
    II (pp. 45-64). Springer, London.

    Parameters
    ----------
    decomposition_method : str, optional
        One of the supported dimentionality reduction methods used as kernel
        for the kM-PDDP algorithm.
    max_clusters_number : int, optional
        Desired maximum number of clusters for the algorithm.
    min_sample_split : int, optional
        The minimum number of points needed in a cluster for a split to occur.
    visualization_utility : bool, optional
        If (True) generate the data needed by the visualization utilities of
        the package othrerwise, if false the split_visualization and
        interactive_visualization of the package can not be created.
    random_seed : int, optional
        The random sedd fed in the k-Means algorithm
    **decomposition_args :
        Arguments for each of the decomposition methods ("PCA" as "pca",
        "KernelPCA" as "kpca", "FastICA" as "ica") utilized by the HiPart
        package, as documented in the scikit-learn package, from which they are
        implemented.

    Attributes
    ----------
    output_matrix : numpy.ndarray
        Model's step by step execution output.
    labels_ :
        Extracted clusters from the algorithm.

    """

    def __init__(
        self,
        decomposition_method="pca",
        max_clusters_number=100,
        min_sample_split=15,
        visualization_utility=True,
        random_seed=None,
        **decomposition_args
    ):
        self.decomposition_method = decomposition_method
        self.max_clusters_number = max_clusters_number
        self.min_sample_split = min_sample_split
        self.visualization_utility = visualization_utility
        self.random_seed = random_seed
        self.decomposition_args = decomposition_args

    def fit(self, X):
        """
        Execute the kM-PDDP algorithm and return all the execution data in the
        form of a kM_PDDP class object.

        Parameters
        ----------
        X: numpy.ndarray
            Data matrix with the samples on the rows and the avriables on the
            columns.

        Returns
        -------
        self : object
            A kM-PDDP class type object, with complete results on the
            algorithm's analysis.

        """
        self.X = X
        self.samples_number = X.shape[0]

        # create an id vector for the samples of X
        indices = np.array([int(i) for i in range(np.size(self.X, 0))])

        # initialize tree and root node                         # step (0)
        tree = Tree()
        # nodes unique IDs indicator
        self.node_ids = 0
        # nodes next color indicator (necessary for visualization purposes)
        self.cluster_color = 0
        tree.create_node(
            tag="cl_" + str(self.node_ids),
            identifier=self.node_ids,
            data=self.calculate_node_data(indices, self.X, self.cluster_color),
        )
        # inidcator for the next node to split
        selected_node = 0

        # if no possibility of split exists on the data     # (ST2)
        if not tree.get_node(0).data["split_permition"]:
            print("cannot split at all")
            return self

        # Initialize the ST1 stopping critirion counter that count the number
        # of clusters
        found_clusters = 1
        # (ST1) or (ST2)
        while (
            (selected_node is not None)
            and (found_clusters < self.max_clusters_number)
        ):

            self.split_function(tree, selected_node)  # step (1)

            # select the next kid for split based on the local minimum density
            selected_node = self.select_kid(tree.leaves())  # step (2)
            found_clusters = found_clusters + 1  # (ST1)

        self.tree = tree
        return self

    def fit_predict(self, X):
        """
        Execute the kM-PDDP algorithm and return the results of the execution
        in the form of labels.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix with the samples on the rows and the avriables on the
            columns.

        Returns
        -------
        labels_ : numpy.ndarray
            Extracted clusters from the algorithm.

        """

        return self.fit(X).labels_

    def split_function(self, tree, selected_node):
        """
        Split the indicated node based on the binary k-Means clustering of the
        node's data projected on one dimention with the dicomposition method.

        Because python passes by refference data this function doesn't need a
        return statment.

        Parameters
        ----------
        tree : treelib.tree.Tree
            The tree build by the kM_PDDP algorithm, in order to cluster the
            input data.

        Returns
        -------
            There no returns in this function. The results of this funciton
            pass to execution by utilizing the python's pass-by-reference
            nature.

        """

        node = tree.get_node(selected_node)
        node.data["split_permition"] = False

        # left child indecies extracted from the nodes splitpoint and the
        # indecies included in the parent node
        left_kid_index = node.data["left_indeces"]
        # right child indecies
        right_kid_index = node.data["right_indeces"]

        # Nodes and data creation for the children
        # Uses the calculate_node_data function to create the data for the node
        tree.create_node(
            tag="cl" + str(self.node_ids + 1),
            identifier=self.node_ids + 1,
            parent=node.identifier,
            data=self.calculate_node_data(
                left_kid_index,
                self.X[left_kid_index, :],
                node.data["color_key"],
            ),
        )
        tree.create_node(
            tag="cl" + str(self.node_ids + 2),
            identifier=self.node_ids + 2,
            parent=node.identifier,
            data=self.calculate_node_data(
                right_kid_index,
                self.X[right_kid_index, :],
                self.cluster_color + 1,
            ),
        )

        self.cluster_color += 1
        self.node_ids += 2

    def select_kid(self, leaves):
        """
        The clusters each time exist in the leaves of the trees. From those
        leaves select the next leave to split based on the algorithm's
        specifications.

        This function creates the nescesary cause for the stopping criterion
        ST1.

        Parameters
        ----------
        leaves : list of treelib.node.Node
            The list of nodes needed to exam to select the next Node to split.

        Returns
        -------
        next_split : int
            The identifier of the next node to split by the algorithm.

        """

        next_split = None

        # Remove the nodes that can not split further
        leaves = list(
            np.array(leaves)[
                [
                    True if i.data["split_criterion"] is not None else False
                    for i in leaves
                ]
            ]
        )

        if len(leaves) > 0:
            for i in sorted(
                enumerate(leaves),
                key=lambda x: x[1].data["split_criterion"],
                reverse=True,
            ):
                if i[1].data["split_permition"]:
                    next_split = i[1].identifier
                    break

        return next_split

    def calculate_node_data(self, indices, data_matrix, key):
        """
        Calculation of the projections onto the Principal Components with the
        utilization of the "Principal Components Analysis" or the "Kernel
        Principal Components Analysis" or the "Indipented Component Analysis"
        methods.

        Determination of the projection's clusters by utilizing the binary
        k-means clustering algorithm.

        This function leads to the second Stopping criterion 2 of the
        algorithm.

        Parameters
        ----------
        indices : numpy.ndarray
            The index of the samples in the original data matrix.
        data_matrix : numpy.ndarray
            The data matrix containing all the data for the samples.
        key : int
            The value of the color for each node.

        Returns
        -------
        data : dict
            The necesary data for each node which are spliting point

        """
        # if the number of samples
        if indices.shape[0] > self.min_sample_split:
            # execute pca on the data matrix
            projection = util.execute_decomposition_method(
                data_matrix=data_matrix,
                decomposition_method=self.decomposition_method,
                two_dimentions=self.visualization_utility,
                decomposition_args=self.decomposition_args,
            )
            one_dimention = np.array([[i] for i in projection[:, 0]])

            model = KMeans(n_clusters=2, random_state=self.random_seed)
            model.fit(one_dimention)
            labels = model.predict(one_dimention)
            centers = model.cluster_centers_

            # Labels for the split selection
            label_zero = np.where(labels == 0)
            label_one = np.where(labels == 1)

            # The indices of the
            left_child = indices[label_zero]
            right_child = indices[label_one]

            # Total scatter value calculation for the selection of the next
            # cluster to split.
            centered = util.center_data(data_matrix)
            split_criterion = np.linalg.norm(centered, ord="fro")

            right_min = np.min(one_dimention[label_one])
            left_max = np.max(one_dimention[label_zero])
            if left_max > right_min:
                right_min = np.min(one_dimention[label_zero])
                left_max = np.max(one_dimention[label_one])

            splitpoint = left_max + ((right_min - left_max) / 2)
            flag = True
        # =========================
        else:
            left_child = None  # (ST2)
            right_child = None  # (ST2)
            projection = None  # (ST2)
            centers = None  # (ST2)
            splitpoint = None  # (ST2)
            split_criterion = None  # (ST2)
            flag = False  # (ST2)

        return {
            "indices": indices,
            "left_indeces": left_child,
            "right_indeces": right_child,
            "projection": projection,
            "centers": centers,
            "splitpoint": splitpoint,
            "split_criterion": split_criterion,
            "split_permition": flag,
            "color_key": key,
            "dendrogram_check": False,
        }

    @property
    def decomposition_method(self):
        return self._decomposition_method

    @decomposition_method.setter
    def decomposition_method(self, v):
        if not (v in ["pca", "kpca", "ica"]):
            raise ValueError(
                "decomposition_method: "
                + str(v)
                + ": Unknown decomposition method!"
            )
        self._decomposition_method = v

    @property
    def max_clusters_number(self):
        return self._max_clusters_number

    @max_clusters_number.setter
    def max_clusters_number(self, v):
        if v < 0 or (not isinstance(v, int)):
            raise ValueError("min_sample_split: "
                             + "Invalid value it should be int and > 1")
        self._max_clusters_number = v

    @property
    def min_sample_split(self):
        return self._min_sample_split

    @min_sample_split.setter
    def min_sample_split(self, v):
        if v < 0 or (not isinstance(v, int)):
            raise ValueError("min_sample_split: "
                             + "Invalid value it should be int and > 1")
        self._min_sample_split = v

    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, v):
        if v is not None and (not isinstance(v, int)):
            raise ValueError("min_sample_split: "
                             + "Invalid value it should be int and > 1")
        self._random_seed = v

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, v):
        self._tree = v

    @property
    def output_matrix(self):
        ndDict = self.tree.nodes
        output_matrix = [np.zeros(np.size(self.X, 0))]
        for i in ndDict:
            if not ndDict[i].is_leaf():
                # create output cluster spliting matrix
                tmp = np.copy(output_matrix[-1])
                tmp[
                    self.tree.children(i)[0].data["indices"]
                ] = self.tree.children(i)[0].identifier
                tmp[
                    self.tree.children(i)[1].data["indices"]
                ] = self.tree.children(i)[1].identifier
                output_matrix.append(tmp)
        del output_matrix[0]
        output_matrix = np.array(output_matrix).transpose()
        self.output_matrix = output_matrix
        return self._output_matrix

    @output_matrix.setter
    def output_matrix(self, v):
        self._output_matrix = v

    @property
    def labels_(self):
        labels_ = np.ones(np.size(self.X, 0))
        for i in self.tree.leaves():
            labels_[i.data["indices"]] = i.identifier
        self.labels_ = labels_
        return self._labels_

    @labels_.setter
    def labels_(self, v):
        self._labels_ = v


class PDDP:
    """
    Class PDDP. It executes the PDDP algorithm.

    References
    ----------
    Boley, D. (1998). Principal direction divisive partitioning. Data mining
    and knowledge discovery, 2(4), 325-344.

    Parameters
    ----------
    decomposition_method : str, optional
        One of the supported dimentionality reduction methods used as kernel
        for the PDDP algorithm.
    max_clusters_number : int, optional
        Desired maximum number of clusters for the algorithm.
    min_sample_split : int, optional
        The minimum number of points needed in a cluster for a split to occur.
    visualization_utility : bool, optional
        If (True) generate the data needed by the visualization utilities of
        the package othrerwise, if false the split_visualization and
        interactive_visualization of the package can not be created.
    **decomposition_args :
        Arguments for each of the decomposition methods ("PCA" as "pca",
        "KernelPCA" as "kpca", "FastICA" as "ica") utilized by the HiPart
        package, as documented in the scikit-learn package, from which they are
        implemented.

    Attributes
    ----------
    output_matrix : numpy.ndarray
        Model's step by step execution output.
    labels_ :
        Extracted clusters from the algorithm.

    """

    def __init__(
        self,
        decomposition_method="pca",
        max_clusters_number=100,
        min_sample_split=5,
        visualization_utility=True,
        **decomposition_args
    ):
        self.decomposition_method = decomposition_method
        self.max_clusters_number = max_clusters_number
        self.min_sample_split = min_sample_split
        self.visualization_utility = visualization_utility
        self.decomposition_args = decomposition_args

    def fit(self, X):
        """
        Execute the PDDP algorithm and return all the execution data in the
        form of a PDDP class object.

        Parameters
        ----------
        X: numpy.ndarray
            Data matrix (must check and return an error if not).

        Returns
        -------
        self
            A PDDP class type object, with complete results on the algorithm's
            analysis.

        """

        self.X = X
        self.samples_number = X.shape[0]

        # create an id vector for the samples of X
        indices = np.array([int(i) for i in range(np.size(self.X, 0))])

        # initialize tree and root node                         # step (0)
        tree = Tree()
        # nodes unique IDs indicator
        self.node_ids = 0
        # nodes next color indicator (necessary for visualization purposes)
        self.cluster_color = 0
        tree.create_node(
            tag="cl_" + str(self.node_ids),
            identifier=self.node_ids,
            data=self.calculate_node_data(indices, self.X, self.cluster_color),
        )
        # inidcator for the next node to split
        selected_node = 0

        # if no possibility of split exists on the data     # (ST2)
        if not tree.get_node(0).data["split_permition"]:
            print("cannot split at all")
            return self

        # Initialize the ST1 stopping critirion counter that count the number
        # of clusters                                       # (ST1)
        found_clusters = 1
        while (
            (selected_node is not None)
            and (found_clusters < self.max_clusters_number)
        ):  # (ST1) or (ST2)

            self.split_function(tree, selected_node)  # step (1)

            # select the next kid for split based on the local minimum density
            selected_node = self.select_kid(tree.leaves())  # step (2)
            found_clusters = found_clusters + 1  # (ST1)

        self.tree = tree
        return self

    def fit_predict(self, X):
        """
        Execute the PDDP algorithm and return the results of the execution in
        the form of labels.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix with the samples on the rows and the avriables on the
            columns.

        Returns
        -------
        labels_ : numpy.ndarray
            Extracted clusters from the algorithm.

        """

        return self.fit(X).labels_

    def split_function(self, tree, selected_node):
        """
        Split the indicated node on the minimum of the median of the data
        projected on the first principal component.

        Because python passes by refference data this function doesn't need a
        return statment.

        Parameters
        ----------
        tree : treelib.tree.Tree
            The tree build by the PDDP algorithm, in order to cluster the
            input data.

        Returns
        -------
            There no returns in this function. The results of this funciton
            pass to execution by utilizing the python's pass-by-reference
            nature.

        """

        node = tree.get_node(selected_node)
        node.data["split_permition"] = False

        # left child indecies extracted from the nodes splitpoint and the
        # indecies included in the parent node
        left_kid_index = node.data["indices"][
            np.where(
                node.data["projection"][:, 0] < node.data["splitpoint"]
            )[0]
        ]
        # right child indecies
        right_kid_index = node.data["indices"][
            np.where(
                node.data["projection"][:, 0] >= node.data["splitpoint"]
            )[0]
        ]

        # Nodes and data creation for the children
        # Uses the calculate_node_data function to create the data for the node
        tree.create_node(
            tag="cl" + str(self.node_ids + 1),
            identifier=self.node_ids + 1,
            parent=node.identifier,
            data=self.calculate_node_data(
                left_kid_index,
                self.X[left_kid_index, :],
                node.data["color_key"],
            ),
        )
        tree.create_node(
            tag="cl" + str(self.node_ids + 2),
            identifier=self.node_ids + 2,
            parent=node.identifier,
            data=self.calculate_node_data(
                right_kid_index,
                self.X[right_kid_index, :],
                self.cluster_color + 1,
            ),
        )

        self.cluster_color += 1
        self.node_ids += 2

    def select_kid(self, leaves):
        """
        The clusters each time exist in the leaves of the trees. From those
        leaves select the next leave to split based on the algorithm's
        specifications.

        This function creates the nescesary cause for the stopping criterion
        ST1.

        Parameters
        ----------
        leaves : list of treelib.node.Node
            The list of nodes needed to exam to select the next Node to split.

        Returns
        -------
        int
            A PDDP class type object, with complete results on the algorithm's
            analysis.

        """

        minimum_location = None

        # Remove the nodes that can not split further
        leaves = list(
            np.array(leaves)[
                [
                    True if i.data["split_criterion"] is not None else False
                    for i in leaves
                ]
            ]
        )

        if len(leaves) > 0:
            for i in sorted(
                enumerate(leaves),
                key=lambda x: x[1].data["split_criterion"],
                reverse=True,
            ):
                if i[1].data["split_permition"]:
                    minimum_location = i[1].identifier
                    break

        return minimum_location

    def calculate_node_data(self, indices, data_matrix, key):
        """
        Calculation of the projections onto the Principal Components with the
        utilization of the "Principal Components Analysis" or the "Kernel
        Principal Components Analysis" or the "Indipented Component Analysis"
        methods.

        The projection's clusters are split on the median pf the projected
        data.

        This function leads to the second Stopping criterion 2 of the
        algorithm.

        Parameters
        ----------
        indices : numpy.ndarray
            The index of the samples in the original data matrix.

        data_matrix : numpy.ndarray
            The data matrix containing all the data for the samples.

        key : int
            The value of the color for each node.

        Returns
        -------
        data : dictionary
            The necesary data for each node which are spliting point.

        """

        # if the number of samples
        if indices.shape[0] > self.min_sample_split:

            centered = util.center_data(data_matrix)

            # execute pca on the data matrix
            projection = util.execute_decomposition_method(
                data_matrix=centered,
                decomposition_method=self.decomposition_method,
                two_dimentions=self.visualization_utility,
                decomposition_args=self.decomposition_args,
            )

            # Total scatter value calculation for the selection of the next
            # cluster to split.
            scat = np.linalg.norm(centered, ord="fro")

            if not np.allclose(scat, 0):
                splitpoint = 0.0
                split_criterion = scat
                flag = True
            else:
                splitpoint = None  # (ST2)
                split_criterion = None  # (ST2)
                flag = False  # (ST2)
        # =========================
        else:
            projection = None
            splitpoint = None  # (ST2)
            split_criterion = None  # (ST2)
            flag = False  # (ST2)

        return {
            "indices": indices,
            "projection": projection,
            "splitpoint": splitpoint,
            "split_criterion": split_criterion,
            "split_permition": flag,
            "color_key": key,
            "dendrogram_check": False,
        }

    @property
    def decomposition_method(self):
        return self._decomposition_method

    @decomposition_method.setter
    def decomposition_method(self, v):
        if not (v in ["pca", "kpca", "ica"]):
            raise ValueError(
                "decomposition_method: "
                + str(v)
                + ": Unknown decomposition method!"
            )
        self._decomposition_method = v

    @property
    def max_clusters_number(self):
        return self._max_clusters_number

    @max_clusters_number.setter
    def max_clusters_number(self, v):
        if v < 0 or (not isinstance(v, int)):
            raise ValueError("min_sample_split: "
                             + "Invalid value it should be int and > 1")
        self._max_clusters_number = v

    @property
    def min_sample_split(self):
        return self._min_sample_split

    @min_sample_split.setter
    def min_sample_split(self, v):
        if v < 0 or (not isinstance(v, int)):
            raise ValueError("min_sample_split: "
                             + "Invalid value it should be int and > 1")
        self._min_sample_split = v

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, v):
        self._tree = v

    @property
    def output_matrix(self):
        ndDict = self.tree.nodes
        output_matrix = [np.zeros(np.size(self.X, 0))]
        for i in ndDict:
            if not ndDict[i].is_leaf():
                # create output cluster spliting matrix
                tmp = np.copy(output_matrix[-1])
                tmp[
                    self.tree.children(i)[0].data["indices"]
                ] = self.tree.children(i)[0].identifier
                tmp[
                    self.tree.children(i)[1].data["indices"]
                ] = self.tree.children(i)[1].identifier
                output_matrix.append(tmp)
        del output_matrix[0]
        output_matrix = np.array(output_matrix).transpose()
        self.output_matrix = output_matrix
        return self._output_matrix

    @output_matrix.setter
    def output_matrix(self, v):
        self._output_matrix = v

    @property
    def labels_(self):
        labels_ = np.ones(np.size(self.X, 0))
        for i in self.tree.leaves():
            labels_[i.data["indices"]] = i.identifier
        self.labels_ = labels_
        return self._labels_

    @labels_.setter
    def labels_(self, v):
        self._labels_ = v


class bicecting_kmeans:
    """
    Class bicecting_kmeans. It executes the bisectiong k-Means algorithm.

    References
    ----------
    Savaresi, S. M., & Boley, D. L. (2001, April). On the performance of
    bisecting K-means and PDDP. In Proceedings of the 2001 SIAM International
    Conference on Data Mining (pp. 1-14). Society for Industrial and Applied
    Mathematics.

    Parameters
    ----------
    max_clusters_number : int, optional
        Desired maximum number of clusters for the algorithm.
    min_sample_split : int, optional
        The minimum number of points needed in a cluster for a split to occur.
    random_seed : int, optional
        The random sedd fed in the k-Means algorithm.

    Attributes
    ----------
    output_matrix : numpy.ndarray
        Model's step by step execution output.
    labels_ : numpy.ndarray
        Extracted clusters from the algorithm.

    """

    def __init__(
        self,
        max_clusters_number=100,
        min_sample_split=5,
        random_seed=None
    ):
        self.max_clusters_number = max_clusters_number
        self.min_sample_split = min_sample_split
        self.random_seed = random_seed

    def fit(self, X):
        """
        Execute the bicecting_kmeans algorithm and return all the execution
        data in the form of a bicecting_kmeans class object.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix with the samples on the rows and the avriables on the
            columns.

        Returns
        -------
        self
            A bicecting_kmeans class type object, with complete results on the
            algorithm's analysis.

        """
        self.X = X
        self.samples_number = X.shape[0]

        # create an id vector for the samples of X
        indices = np.array([int(i) for i in range(np.size(self.X, 0))])

        # initialize tree and root node                         # step (0)
        tree = Tree()
        # nodes` unique IDs indicator
        self.node_ids = 0
        # nodes` next color indicator (necessary for visualization purposes)
        self.cluster_color = 0
        tree.create_node(
            tag="cl_" + str(self.node_ids),
            identifier=self.node_ids,
            data=self.calculate_node_data(indices, self.X, self.cluster_color),
        )
        # inidcator for the next node to split
        selected_node = 0

        # if no possibility of split exists on the data
        if not tree.get_node(0).data["split_permition"]:
            print("cannot split at all")
            return self

        # Initialize the ST1 stopping critirion counter that count the number
        # of clusters.
        found_clusters = 1
        while (
            (selected_node is not None)
            and (found_clusters < self.max_clusters_number)
        ):

            self.split_function(tree, selected_node)  # step (1)

            # select the next kid for split based on the local minimum density
            selected_node = self.select_kid(tree.leaves())  # step (2)
            found_clusters = found_clusters + 1  # (ST1)

        self.tree = tree
        return self

    def fit_predict(self, X):
        """
        Execute the bicecting_kmeans algorithm and return the results of the
        execution in the form of labels.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix with the samples on the rows and the avriables on the
            columns.

        Returns
        -------
        labels_ : numpy.ndarray
            Extracted clusters from the algorithm.

        """

        return self.fit(X).labels_

    def split_function(self, tree, selected_node):
        """
        Split the indicated node by clustering the data with a binary k-menas
        clustering algorithm.

        Because python passes by refference data this function doesn't need a
        return statment.

        Parameters
        ----------
        tree : treelib.tree.Tree
            The tree build by the bicecting_kmeans algorithm, in order to
            cluster the input data.

        Returns
        -------
            There is no returns in this function. The results of this funciton
            pass to execution by utilizing the python's pass-by-reference
            nature.

        """
        node = tree.get_node(selected_node)
        node.data["split_permition"] = False

        # left child indecies extracted from the nodes splitpoint and the
        # indecies included in the parent node
        left_kid_index = node.data["left_indeces"]

        # right child indecies
        right_kid_index = node.data["right_indeces"]

        # Nodes and data creation for the children
        # Uses the calculate_node_data function to create the data for the node
        tree.create_node(
            tag="cl" + str(self.node_ids + 1),
            identifier=self.node_ids + 1,
            parent=node.identifier,
            data=self.calculate_node_data(
                left_kid_index,
                self.X[left_kid_index, :],
                node.data["color_key"],
            ),
        )
        tree.create_node(
            tag="cl" + str(self.node_ids + 2),
            identifier=self.node_ids + 2,
            parent=node.identifier,
            data=self.calculate_node_data(
                right_kid_index,
                self.X[right_kid_index, :],
                self.cluster_color + 1,
            ),
        )

        self.cluster_color += 1
        self.node_ids += 2

    def select_kid(self, leaves):
        """
        The clusters each time exist in the leaves of the trees. From those
        leaves select the next leave to split based on the algorithm's
        specifications.

        This function creates the nescesary data for further execution of the
        algorithm.

        Parameters
        ----------
        leaves : list of treelib.node.Node
            The list of nodes needed to exam to select the next Node to split.

        Returns
        -------
        next_split : int
            The identifier of the next node to split by the algorithm.

        """
        next_split = None

        # Remove the nodes that can not split further
        leaves = list(
            np.array(leaves)[
                [
                    True if i.data["split_criterion"] is not None else False
                    for i in leaves
                ]
            ]
        )

        if len(leaves) > 0:
            for i in sorted(
                enumerate(leaves),
                key=lambda x: x[1].data["split_criterion"],
                reverse=True,
            ):
                if i[1].data["split_permition"]:
                    next_split = i[1].identifier
                    break

        return next_split

    def calculate_node_data(self, indices, data_matrix, key):
        """
        Execution of the binary k-Means algorithm on the samples presented by
        the data_matrix. The two resulted clusters are the two new clusters if
        the leaf is chosen to be splited. And calculation of the spliting
        criterion.

        Parameters
        ----------
        indices : numpy.ndarray
            The index of the samples in the original data matrix.
        data_matrix : numpy.ndarray
            The data matrix containing all the data for the samples.
        key : int
            The value of the color for each node.

        Returns
        -------
        data : dict
            The necesary data for each node which are spliting point.

        """
        # if the number of samples
        if indices.shape[0] > self.min_sample_split:

            model = KMeans(n_clusters=2, random_state=self.random_seed)
            model.fit(data_matrix)
            labels = model.predict(data_matrix)
            centers = model.cluster_centers_

            left_child = indices[np.where(labels == 0)]
            right_child = indices[np.where(labels == 1)]
            centers = centers

            centered = util.center_data(data_matrix)
            # Total scatter value calculation for the selection of the next
            # cluster to split.
            scat = np.linalg.norm(centered, ord="fro")

            split_criterion = scat
            flag = True
        # =========================
        else:
            left_child = None  # (ST2)
            right_child = None  # (ST2)
            centers = None  # (ST2)
            split_criterion = None  # (ST2)
            flag = False  # (ST2)

        return {
            "indices": indices,
            "left_indeces": left_child,
            "right_indeces": right_child,
            "centers": centers,
            "split_criterion": split_criterion,
            "split_permition": flag,
            "color_key": key,
            "dendrogram_check": False,
        }

    @property
    def max_clusters_number(self):
        return self._max_clusters_number

    @max_clusters_number.setter
    def max_clusters_number(self, v):
        if v < 0 or (not isinstance(v, int)):
            raise ValueError("min_sample_split: "
                             + "Invalid value it should be int and > 1")
        self._max_clusters_number = v

    @property
    def min_sample_split(self):
        return self._min_sample_split

    @min_sample_split.setter
    def min_sample_split(self, v):
        if v < 0 or (not isinstance(v, int)):
            raise ValueError("min_sample_split: "
                             + "Invalid value it should be int and > 1")
        self._min_sample_split = v

    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, v):
        if v is not None and (not isinstance(v, int)):
            raise ValueError("min_sample_split: "
                             + "Invalid value it should be int and > 1")
        self._random_seed = v

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, v):
        self._tree = v

    @property
    def output_matrix(self):
        ndDict = self.tree.nodes
        output_matrix = [np.zeros(np.size(self.X, 0))]
        for i in ndDict:
            if not ndDict[i].is_leaf():
                # create output cluster spliting matrix
                tmp = np.copy(output_matrix[-1])
                tmp[
                    self.tree.children(i)[0].data["indices"]
                ] = self.tree.children(i)[0].identifier
                tmp[
                    self.tree.children(i)[1].data["indices"]
                ] = self.tree.children(i)[1].identifier
                output_matrix.append(tmp)
        del output_matrix[0]
        output_matrix = np.array(output_matrix).transpose()
        self.output_matrix = output_matrix
        return self._output_matrix

    @output_matrix.setter
    def output_matrix(self, v):
        self._output_matrix = v

    @property
    def labels_(self):
        labels_ = np.ones(np.size(self.X, 0))
        for i in self.tree.leaves():
            labels_[i.data["indices"]] = i.identifier
        self.labels_ = labels_
        return self._labels_

    @labels_.setter
    def labels_(self, v):
        self._labels_ = v
