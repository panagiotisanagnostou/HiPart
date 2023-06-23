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
Implementation of the clustering algorithms, members of the HiPart package.

@author Panagiotis Anagnostou
@author Nicos Pavlidis
"""

import HiPart.__utility_functions as util
import numpy as np
import statsmodels.api as sm
import warnings

from KDEpy import FFTKDE
from scipy import stats
from sklearn.cluster import KMeans
from treelib import Tree


class DePDDP:
    """
    Class dePDDP. It executes the dePDDP algorithm.

    References
    ----------
    Tasoulis, S. K., Tasoulis, D. K., & Plagianakos, V. P. (2010). Enhancing
    principal direction divisive clustering. Pattern Recognition, 43(10), 3391-
    3411.

    Parameters
    ----------
    decomposition_method : str, (optional)
        One of the ('pca', 'kpca', 'ica', 'tsne') supported decomposition
        methods used as kernel for the dePDDP algorithm.
    max_clusters_number : int, (optional)
        Desired maximum number of clusters to find the dePDDP algorithm.
    bandwidth_scale : float, (optional)
        Standard deviation scaler for the density approximation.
    percentile : float, (optional)
        The percentile distance from the dataset's edge in which a split can
        not occur. [0,0.5) values are allowed.
    min_sample_split : int, (optional)
        The minimum number of points needed in a cluster for a split to occur.
    visualization_utility : bool, (optional)
        If (True) generate the data needed by the visualization utilities of
        the package otherwise, if false the split_visualization and
        interactive_visualization of the package can not be created. For the
        'tsne' decomposition method does not support visualization because it
        affects the correct execution of the dePDDP algorithm.
    **decomposition_args :
        Arguments for each of the decomposition methods ("decomposition.PCA" as
        "pca", "decomposition.KernelPCA" as "kpca", "decomposition.FastICA" as
        "ica", "manifold.TSNE" as "tsne") utilized by the HiPart package, as
        documented in the scikit-learn package, from which they are implemented.

    Attributes
    ----------
    output_matrix : numpy.ndarray
        Model's step by step execution output.
    labels_ : numpy.ndarray
        Extracted clusters from the algorithm.
    tree : treelib.Tree
        The object which contains all the information about the execution of
        the dePDDP algorithm.
    samples_number : int
        The number of samples contained in the data.

    """

    def __init__(
        self,
        decomposition_method="pca",
        max_clusters_number=100,
        bandwidth_scale=0.5,
        percentile=0.1,
        min_sample_split=5,
        visualization_utility=True,
        **decomposition_args,
    ):
        self.decomposition_method = decomposition_method
        self.max_clusters_number = max_clusters_number
        self.bandwidth_scale = bandwidth_scale
        self.percentile = percentile
        self.min_sample_split = min_sample_split
        if decomposition_method in ["tsne"]:
            self.visualization_utility = False
            warnings.warn("DePDDP: does not support visualization for 'tsne'.")
        else:
            self.visualization_utility = visualization_utility
        self.decomposition_args = decomposition_args

    def fit(self, X):
        """
        Execute the dePDDP algorithm and return all the execution data in the
        form of a dePDDP class object.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix with the samples on the rows and the variables on the
            columns.

        Returns
        -------
        self
            A dePDDP class type object, with complete results on the
            algorithm's analysis.

        """
        self.X = X
        self.samples_number = np.size(X, 0)

        # create an id vector for the samples of X
        indices = np.array([int(i) for i in range(self.samples_number)])

        # initialize the tree and root node                           # step (0)
        den_tree = Tree()
        # nodes unique IDs indicator
        self.node_ids = 0
        # nodes next color indicator (necessary for visualization purposes)
        self.cluster_color = 0
        den_tree.create_node(
            tag="density_cluster_" + str(self.node_ids),
            identifier=self.node_ids,
            data=self.calculate_node_data(indices, self.cluster_color),
        )
        # indicator for the next node to split
        selected_node = 0

        # if no possibility of split exists on the data                  # (ST2)
        if not den_tree.get_node(0).data["split_permission"]:
            raise RuntimeError("dePDDP cannot split the data at all!!!")

        # Initialize the ST1 stopping criterion counter that count the number
        # of clusters                                                    # (ST1)
        found_clusters = 1
        while (found_clusters < self.max_clusters_number) and (
            selected_node is not None
        ):  # (ST1) or (ST2)

            self.split_function(den_tree, selected_node)  # step (1, 2)

            # select the next kid for split based on the local minimum density
            selected_node = self.select_kid(den_tree.leaves())  # step (3)
            found_clusters = found_clusters + 1  # (ST1)

        self.tree = den_tree
        return self

    def fit_predict(self, X):
        """
        Execute the dePDDP algorithm and return the results of the execution
        in the form of labels.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix with the samples on the rows and the variables on the
            columns.

        Returns
        -------
        labels_ : numpy.ndarray
            Extracted clusters from the algorithm.

        """

        return self.fit(X).labels_

    def split_function(self, den_tree, selected_node):
        """
        Split the indicated node on the minimum of the local minimum density
        of the data projected on the first principal component.

        Because python passes by reference data this function doesn't need a
        return statement.

        Parameters
        ----------
        den_tree : treelib.tree.Tree
            The tree build by the dePDDP algorithm, in order to cluster the
            input data.
        selected_node : int
            The numerical identifier for the tree node that i about to be split.

        Returns
        -------
            There no returns in this function. The results of this function
            pass to execution by utilizing the python's pass-by-reference
            nature.

        """
        node = den_tree.get_node(selected_node)
        node.data["split_permission"] = False

        # left child indices extracted from the nodes split-point and the
        # indices included in the parent node
        left_kid_index = node.data["indices"][
            np.where(node.data["projection"][:, 0] < node.data["splitpoint"])[0]
        ]
        # right child indices
        right_kid_index = node.data["indices"][
            np.where(node.data["projection"][:, 0] >= node.data["splitpoint"])[0]
        ]

        # Nodes and data creation for the children
        # Uses the calculate_node_data function to create the data for the node
        den_tree.create_node(
            tag="density_cluster_" + str(self.node_ids + 1),
            identifier=self.node_ids + 1,
            parent=node.identifier,
            data=self.calculate_node_data(left_kid_index, node.data["color_key"]),
        )
        den_tree.create_node(
            tag="density_cluster_" + str(self.node_ids + 2),
            identifier=self.node_ids + 2,
            parent=node.identifier,
            data=self.calculate_node_data(right_kid_index, self.cluster_color + 1),
        )

        self.cluster_color += 1
        self.node_ids += 2

    def select_kid(self, possible_splits):
        """
        The clusters each time exist in the leaves of the trees. From those
        leaves select the next leave to split based on the algorithm's
        specifications.

        This function creates the necessary cause for the stopping criterion
        ST1.

        Parameters
        ----------
        possible_splits : list of treelib.node.Node
            The list of nodes needed to exam to select the next Node to split.

        Returns
        -------
        next_split : int
            The identifier of the next node to split by the algorithm.

        """
        min_density_node = None

        # Remove the nodes that can not split further
        possible_splits = list(
            np.array(possible_splits)[
                [
                    True if i.data["split_criterion"] is not None else False
                    for i in possible_splits
                ]
            ]
        )

        if len(possible_splits) > 0:
            for i in sorted(
                enumerate(possible_splits), key=lambda x: x[1].data["split_criterion"]
            ):
                if i[1].data["split_permission"]:
                    min_density_node = i[1].identifier
                    break

        return min_density_node

    def calculate_node_data(self, indices, key):
        """
        Calculation of the projections onto the Principal Components with the
        utilization of the "Principal Components Analysis" or the "Kernel
        Principal Components Analysis" or the "Independent Component Analysis"
        or "t-SNE" methods.

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
        key : int
            The value of the color for each node.

        Returns
        -------
        data : dict
            The necessary data for each node which are splitting point.

        """

        proj = None  # (ST2)
        splitpoint = None  # (ST2)
        split_criterion = None  # (ST2)
        flag = False  # (ST2)

        # if the number of samples
        if indices.shape[0] > self.min_sample_split:
            # execute pca on the data matrix
            proj = util.execute_decomposition_method(
                data_matrix=self.X[indices, :],
                decomposition_method=self.decomposition_method,
                two_dimentions=self.visualization_utility,
                decomposition_args=self.decomposition_args,
            )
            one_dimension = proj[:, 0]

            # calculate the standard deviation of the data
            bandwidth = sm.nonparametric.bandwidths.select_bandwidth(
                one_dimension, "silverman", kernel=None
            )

            # calculate the density function on the 1st Principal Component
            # x_ticks: projection points on the 1st PC
            # evaluation: the density of the projections on the 1st PC
            x_ticks, evaluation = (
                FFTKDE(kernel="gaussian", bw=self.bandwidth_scale * bandwidth)
                .fit(one_dimension)
                .evaluate()
            )
            # calculate all the local minima
            minimum_indices = np.where(
                np.diff((np.diff(evaluation) > 0).astype("int8")) == 1
            )[0]

            # Find the location of the local minima and make sure they are
            # with in the given percentile limits
            quantile_value = np.quantile(
                one_dimension, (self.percentile, (1 - self.percentile))
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

        return {
            "indices": indices,
            "projection": proj,
            "splitpoint": splitpoint,
            "split_criterion": split_criterion,
            "split_permission": flag,
            "color_key": key,
            "dendrogram_check": False,
        }

    @property
    def decomposition_method(self):
        return self._decomposition_method

    @decomposition_method.setter
    def decomposition_method(self, v):
        if not (v in ["pca", "kpca", "ica", "tsne"]):
            raise ValueError(
                "DePDDP: decomposition_method: "
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
            raise ValueError(
                "DePDDP: min_sample_split: Invalid value it should be int and > 1"
            )
        self._max_clusters_number = v

    @property
    def bandwidth_scale(self):
        return self._bandwidth_scale

    @bandwidth_scale.setter
    def bandwidth_scale(self, v):
        if v <= 0:
            raise ValueError("DePDDP: bandwidth_scale: Should be > 0")
        self._bandwidth_scale = v

    @property
    def percentile(self):
        return self._percentile

    @percentile.setter
    def percentile(self, v):
        if v >= 0.5 or v < 0:
            raise ValueError("DePDDP: percentile: Should be between [0,0.5) interval")
        self._percentile = v

    @property
    def min_sample_split(self):
        return self._min_sample_split

    @min_sample_split.setter
    def min_sample_split(self, v):
        if v < 0 or (not isinstance(v, int)):
            raise ValueError(
                "DePDDP: min_sample_split: Invalid value it should be int and > 1"
            )
        self._min_sample_split = v

    @property
    def visualization_utility(self):
        return self._visualization_utility

    @visualization_utility.setter
    def visualization_utility(self, v):
        if v is not True and v is not False:
            raise ValueError(
                "DePDDP: visualization_utility: Should be True or False"
            )

        if v is True and self.decomposition_method not in ["pca", "ica", "kpca"]:
            raise ValueError(
                "DePDDP: visualization_utility: 'tsne' method is can't be used"
                + " with the visualization utility."
            )
        self._visualization_utility = v

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, v):
        self._tree = v

    @property
    def output_matrix(self):
        nd_dict = self.tree.nodes
        output_matrix = [np.zeros(np.size(self.X, 0))]

        # the dictionary of nodes contains the created node from the DePDDP
        # algorithm sorted from the root to the last split
        for i in nd_dict:
            # For the output matrix we don't want the leaves of the tree. Each
            # level of the output matrix represents a split the split exist in
            # the internal nodes of the tree. Only by checking the children of
            # those nodes we can extract the data for the current split.
            if not nd_dict[i].is_leaf():
                # create output cluster splitting matrix
                tmp = np.copy(output_matrix[-1])
                # Left child according to the tree creation process
                tmp[self.tree.children(i)[0].data["indices"]] = self.tree.children(i)[
                    0
                ].identifier
                # Right child according to the tree creation process
                tmp[self.tree.children(i)[1].data["indices"]] = self.tree.children(i)[
                    1
                ].identifier

                # The output_matrix is created transposed
                output_matrix.append(tmp)
        # the first row contains only zeros
        del output_matrix[0]

        # transpose the output_matrix to be extracted
        output_matrix = np.array(output_matrix).transpose()

        return output_matrix

    @output_matrix.setter
    def output_matrix(self, v):
        raise RuntimeError(
            "DePDDP: output_matrix: can only be generated and not to be assigned!"
        )

    @property
    def labels_(self):
        labels_ = np.ones(np.size(self.X, 0))
        for i in self.tree.leaves():
            labels_[i.data["indices"]] = i.identifier
        return labels_

    @labels_.setter
    def labels_(self, v):
        raise RuntimeError(
            "DePDDP: labels_: can only be generated and not to be assigned!"
        )


class IPDDP:
    """
    Class IPDDP. It executes the iPDDP algorithm.

    References
    ----------
    Tasoulis, S. K., Tasoulis, D. K., & Plagianakos, V. P. (2010). Enhancing
    principal direction divisive clustering. Pattern Recognition, 43(10), 3391-
    3411.

    Parameters
    ----------
    decomposition_method : str, (optional)
        One of the ('pca', 'kpca', 'ica', 'tsne') supported decomposition
        methods used as kernel for the iPDDP algorithm.
    max_clusters_number : int, (optional)
        Desired maximum number of clusters for the algorithm.
    percentile : float, (optional)
        The percentile distance from the dataset's edge in which a split can
        not occur. [0,0.5) values are allowed.
    min_sample_split : int, (optional)
        The minimum number of points needed in a cluster for a split to occur.
    visualization_utility : bool, (optional)
        If (True) generate the data needed by the visualization utilities of
        the package otherwise, if false the split_visualization and
        interactive_visualization of the package can not be created. For the
        'tsne' decomposition method does not support visualization because it
        affects the correct execution of the iPDDP algorithm.
    **decomposition_args :
        Arguments for each of the decomposition methods ("decomposition.PCA" as
        "pca", "decomposition.KernelPCA" as "kpca", "decomposition.FastICA" as
        "ica", "manifold.TSNE" as "tsne") utilized by the HiPart package, as
        documented in the scikit-learn package, from which they are implemented.

    Attributes
    ----------
    output_matrix : numpy.ndarray
        Model's step by step execution output.
    labels_ :
        Extracted clusters from the algorithm.
    tree : treelib.Tree
        The object which contains all the information about the execution of
        the iPDDP algorithm.
    samples_number : int
        The number of samples contained in the data.

    """

    def __init__(
        self,
        decomposition_method="pca",
        max_clusters_number=100,
        percentile=0.1,
        min_sample_split=5,
        visualization_utility=True,
        **decomposition_args,
    ):
        self.decomposition_method = decomposition_method
        self.max_clusters_number = max_clusters_number
        self.percentile = percentile
        self.min_sample_split = min_sample_split
        if decomposition_method in ["tsne"]:
            self.visualization_utility = False
            warnings.warn("IPDDP: does not support visualization for 'tsne'.")
        else:
            self.visualization_utility = visualization_utility
        self.decomposition_args = decomposition_args

    def fit(self, X):
        """
        Execute the iPDDP algorithm and return all the execution data in the
        form of a IPDDP class object.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix with the samples on the rows and the variables on the
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
        indices = np.array([int(i) for i in range(X.shape[0])])

        # initialize tree and root node
        proj_tree = Tree()
        # nodes unique IDs indicator
        self.node_ids = 0
        # nodes next color indicator (necessary for visualization purposes)
        self.cluster_color = 0
        # Root initialization
        proj_tree.create_node(  # step (0)
            tag="cl_" + str(self.node_ids),
            identifier=self.node_ids,
            data=self.calculate_node_data(indices, self.cluster_color),
        )
        # indicator for the next node to split
        selected_node = 0

        # if no possibility of split exists on the data
        if not proj_tree.get_node(0).data["split_permission"]:
            raise RuntimeError("iPDDP: cannot split the data at all!!!")

        # Initialize the stopping criterion counter that counts the number
        # of clusters
        splits = 1
        while (selected_node is not None) and (splits < self.max_clusters_number):
            self.split_function(proj_tree, selected_node)  # step (1 ,2)

            # select the next kid for split based on the local minimum density
            selected_node = self.select_kid(proj_tree.leaves())  # step (3)
            splits = splits + 1

        self.tree = proj_tree

        return self

    def fit_predict(self, X):
        """
        Execute the iPDDP algorithm and return the results of the execution
        in the form of labels.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix with the samples on the rows and the variables on the
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

        Because python passes by reference data this function doesn't need a
        return statement.

        Parameters
        ----------
        tree : treelib.tree.Tree
            The tree build by the iPDDP algorithm, in order to cluster the
            input data.
        selected_node : int
            The numerical identifier for the tree node that i about to be split.

        Returns
        -------
            There no returns in this function. The results of this function
            pass to execution by utilizing the python's pass-by-reference
            nature.

        """

        current_node = tree.get_node(selected_node)
        current_node.data["split_permission"] = False

        # left child indices extracted from the nodes split-point and the
        # indices included in the parent node
        left_indices = current_node.data["indices"][
            np.where(
                current_node.data["projection"][:, 0] < current_node.data["splitpoint"]
            )[0]
        ]
        # right child indices
        right_indices = current_node.data["indices"][
            np.where(
                current_node.data["projection"][:, 0] >= current_node.data["splitpoint"]
            )[0]
        ]

        # Nodes and data creation for the children
        # Uses the calculate_node_data function to create the data for the node
        tree.create_node(
            tag="cl" + str(self.node_ids + 1),
            identifier=self.node_ids + 1,
            parent=current_node.identifier,
            data=self.calculate_node_data(
                left_indices,
                current_node.data["color_key"],
            ),
        )
        tree.create_node(
            tag="cl" + str(self.node_ids + 2),
            identifier=self.node_ids + 2,
            parent=current_node.identifier,
            data=self.calculate_node_data(
                right_indices,
                self.cluster_color + 1,
            ),
        )

        self.cluster_color += 1
        self.node_ids += 2

    def select_kid(self, leaves):
        """
        The clusters, in the iPDDP algorithm, exist in the leaves of the trees.
        From those leaves select the next leave to split based on the
        specifications of the algorithm.

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
                if i[1].data["split_permission"]:
                    next_split = i[1].identifier
                    break

        return next_split

    def calculate_node_data(self, indices, key):
        """
        Calculation of the projections onto the Principal Components with the
        utilization of the "Principal Components Analysis" or the "Kernel
        Principal Components Analysis" or the "Independent Component Analysis"
        or "t-SNE" methods.

        Determination of the projection's maximum distance between to
        consecutive points and chooses it as the split-point for this node.

        This function leads to the second Stopping criterion 2 of the
        algorithm.

        Parameters
        ----------
        indices : ndarray of shape (n_samples,)
            The index of the samples in the original data matrix.
        key : int
            The value of the color for each node.

        Returns
        -------
        data : dict
            The necessary data for each node which are splitting point.

        """

        # Application of the minimum sample number split
        # =========================
        if indices.shape[0] > self.min_sample_split:
            # execute pca on the data matrix
            proj_vector = util.execute_decomposition_method(
                data_matrix=self.X[indices, :],
                decomposition_method=self.decomposition_method,
                two_dimentions=self.visualization_utility,
                decomposition_args=self.decomposition_args,
            )
            one_dimension = proj_vector[:, 0]

            sort_indices = np.argsort(one_dimension)
            proj_vector = proj_vector[sort_indices, :]
            one_dimension = proj_vector[:, 0]
            indices = indices[sort_indices]

            quantile_value = np.quantile(
                one_dimension, (self.percentile, (1 - self.percentile))
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
                splitpoint = None
                split_criterion = None
                flag = False
        # =========================
        else:
            proj_vector = None
            splitpoint = None
            split_criterion = None
            flag = False

        return {
            "indices": indices,
            "projection": proj_vector,
            "splitpoint": splitpoint,
            "split_criterion": split_criterion,
            "split_permission": flag,
            "color_key": key,
            "dendrogram_check": False,
        }

    @property
    def max_clusters_number(self):
        return self._max_clusters_number

    @max_clusters_number.setter
    def max_clusters_number(self, v):
        if v < 0 or (not isinstance(v, int)):
            raise ValueError(
                "IPDDP: min_sample_split: Invalid value it should be int and > 1"
            )
        self._max_clusters_number = v

    @property
    def decomposition_method(self):
        return self._decomposition_method

    @decomposition_method.setter
    def decomposition_method(self, v):
        if not (v in ["pca", "kpca", "ica", "tsne"]):
            raise ValueError(
                "IPDDP: decomposition_method: "
                + str(v)
                + ": Unknown decomposition method!"
            )
        self._decomposition_method = v

    @property
    def percentile(self):
        return self._percentile

    @percentile.setter
    def percentile(self, v):
        if v >= 0.5 or v < 0:
            raise ValueError("IPDDP: percentile: Should be between [0,0.5) interval")
        self._percentile = v

    @property
    def min_sample_split(self):
        return self._min_sample_split

    @min_sample_split.setter
    def min_sample_split(self, v):
        if v < 0 or (not isinstance(v, int)):
            raise ValueError(
                "IPDDP: min_sample_split: Invalid value it should be int and > 1"
            )
        self._min_sample_split = v

    @property
    def visualization_utility(self):
        return self._visualization_utility

    @visualization_utility.setter
    def visualization_utility(self, v):
        if v is not True and v is not False:
            raise ValueError(
                "IPDDP: visualization_utility: Should be True or False"
            )

        if v is True and self.decomposition_method not in ["pca", "ica", "kpca"]:
            raise ValueError(
                "IPDDP: visualization_utility: Should be pca when visualization_utility is True."
            )
        self._visualization_utility = v

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, v):
        self._tree = v

    @property
    def output_matrix(self):
        nodes = self.tree.nodes
        output_matrix = [np.zeros(np.size(self.X, 0))]

        # The dictionary of nodes contains the created node from the IPDDP
        # algorithm sorted from the root to the last split.
        for i in nodes:
            # For the output matrix we don't want the leaves of the tree. Each
            # level of the output matrix represents a split the split exist in
            # the internal nodes of the tree. Only by checking the children of
            # those nodes we can extract the data for the current split.
            if not nodes[i].is_leaf():
                # create output cluster splitting matrix
                tmp_array = np.copy(output_matrix[-1])
                # Left child according to the tree creation process
                tmp_array[self.tree.children(i)[0].data["indices"]] = self.tree.children(i)[
                    0
                ].identifier
                # Right child according to the tree creation process
                tmp_array[self.tree.children(i)[1].data["indices"]] = self.tree.children(i)[
                    1
                ].identifier

                # The output_matrix is created transposed
                output_matrix.append(tmp_array)
        # the first row contains only zeros
        del output_matrix[0]

        # transpose the output_matrix to be extracted
        output_matrix = np.array(output_matrix).transpose()

        return output_matrix

    @output_matrix.setter
    def output_matrix(self, v):
        raise RuntimeError(
            "IPDDP: output_matrix: can only be generated and not to be assigned!"
        )

    @property
    def labels_(self):
        labels_ = np.ones(np.size(self.X, 0))
        for i in self.tree.leaves():
            labels_[i.data["indices"]] = i.identifier
        return labels_

    @labels_.setter
    def labels_(self, v):
        raise RuntimeError(
            "IPDDP: labels_: can only be generated and not to be assigned!"
        )


class KMPDDP:
    """
    Class KMPDDP. It executes the kMeans-PDDP algorithm.

    References
    ----------
    Zeimpekis, D., & Gallopoulos, E. (2008). Principal direction divisive
    partitioning with kernels and k-means steering. In Survey of Text Mining
    II (pp. 45-64). Springer, London.

    Parameters
    ----------
    decomposition_method : str, (optional)
        One of the ('pca', 'kpca', 'ica', 'tsne') supported decomposition
        methods used as kernel for the kMeans-PDDP algorithm.
    max_clusters_number : int, (optional)
        Desired maximum number of clusters for the algorithm.
    min_sample_split : int, (optional)
        The minimum number of points needed in a cluster for a split to occur.
    visualization_utility : bool, (optional)
        If (True) generate the data needed by the visualization utilities of
        the package otherwise, if false the split_visualization and
        interactive_visualization of the package can not be created. For the
        'tsne' decomposition method does not support visualization because it
        affects the correct execution of the kMeans-PDDP algorithm.
    random_seed : int, (optional)
        The random seed fed in the k-Means algorithm
    **decomposition_args :
        Arguments for each of the decomposition methods ("decomposition.PCA" as
        "pca", "decomposition.KernelPCA" as "kpca", "decomposition.FastICA" as
        "ica", "manifold.TSNE" as "tsne") utilized by the HiPart package, as
        documented in the scikit-learn package, from which they are implemented.

    Attributes
    ----------
    output_matrix : numpy.ndarray
        Model's step by step execution output.
    labels_ :
        Extracted clusters from the algorithm.
    tree : treelib.Tree
        The object which contains all the information about the execution of
        the iPDDP algorithm.
    samples_number : int
        The number of samples contained in the data.

    """

    def __init__(
        self,
        decomposition_method="pca",
        max_clusters_number=100,
        min_sample_split=15,
        visualization_utility=True,
        random_seed=None,
        **decomposition_args,
    ):
        self.decomposition_method = decomposition_method
        self.max_clusters_number = max_clusters_number
        self.min_sample_split = min_sample_split
        if decomposition_method in ["tsne"]:
            self.visualization_utility = False
            warnings.warn("KMPDDP: does not support visualization for 'tsne'.")
        else:
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
            Data matrix with the samples on the rows and the variables on the
            columns.

        Returns
        -------
        self : object
            A kM-PDDP class type object, with complete results on the
            algorithm's analysis.

        """
        self.X = X
        self.samples_number = np.size(X, 0)

        # create an id vector for the samples of X
        indices = np.arange(self.samples_number)

        # Variable initializations
        bk_tree = Tree()
        self.node_ids = 0
        self.cluster_color = 0
        selected_node = 0
        found = 1

        # Root node initialization
        bk_tree.create_node(
            tag="binary_kmeans_node_" + str(self.node_ids),
            identifier=self.node_ids,
            data=self.calculate_node_data(indices, self.cluster_color),
        )

        # if no possibility of split exists on the data
        if not bk_tree.get_node(0).data["split_permission"]:
            raise RuntimeError("KMPDDP: cannot split the data at all!!!")

        while (selected_node is not None) and (found < self.max_clusters_number):
            # Split the selected node in two parts
            self.split_function(bk_tree, selected_node)

            # select the next kid for split based on the local minimum density
            selected_node = self.select_kid(bk_tree.leaves())

            # every split adds one new cluster
            found = found + 1

        self.tree = bk_tree
        return self

    def fit_predict(self, X):
        """
        Execute the kM-PDDP algorithm and return the results of the execution
        in the form of labels.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix with the samples on the rows and the variables on the
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
        node's data projected on one dimension with the decomposition method.

        Because python passes by reference data this function doesn't need a
        return statement.

        Parameters
        ----------
        tree : treelib.tree.Tree
            The tree build by the kM_PDDP algorithm, in order to cluster the
            input data.
        selected_node : int
            The numerical identifier for the tree node that i about to be split.

        Returns
        -------
            There no returns in this function. The results of this function
            pass to execution by utilizing the python's pass-by-reference
            nature.

        """

        node = tree.get_node(selected_node)
        node.data["split_permission"] = False

        # left child indices extracted from the nodes split-point and the
        # indices included in the parent node
        left_kid = node.data["left_indices"]
        # right child indices
        right_kid = node.data["right_indices"]

        # Nodes and data creation for the children
        # Uses the calculate_node_data function to create the data for the node
        tree.create_node(
            tag="binary_kmeans_node_" + str(self.node_ids + 1),
            identifier=self.node_ids + 1,
            parent=node.identifier,
            data=self.calculate_node_data(
                left_kid,
                node.data["color_key"],
            ),
        )
        tree.create_node(
            tag="binary_kmeans_node_" + str(self.node_ids + 2),
            identifier=self.node_ids + 2,
            parent=node.identifier,
            data=self.calculate_node_data(
                right_kid,
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

        This function creates the necessary cause for the stopping criterion
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

        maximum_frobenius = None

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
                if i[1].data["split_permission"]:
                    maximum_frobenius = i[1].identifier
                    break

        return maximum_frobenius

    def calculate_node_data(self, indices, key):
        """
        Calculation of the projections onto the Principal Components with the
        utilization of the "Principal Components Analysis" or the "Kernel
        Principal Components Analysis" or the "Independent Component Analysis"
        or "t-SNE" methods.

        Determination of the projection's clusters by utilizing the binary
        k-means clustering algorithm.

        This function leads to the second Stopping criterion 2 of the
        algorithm.

        Parameters
        ----------
        indices : numpy.ndarray
            The index of the samples in the original data matrix.
        key : int
            The value of the color for each node.

        Returns
        -------
        data : dict
            The necessary data for each node which are splitting point

        """
        # if the number of samples
        if indices.shape[0] > self.min_sample_split:
            # execute pca on the data matrix
            projection = util.execute_decomposition_method(
                data_matrix=self.X[indices, :],
                decomposition_method=self.decomposition_method,
                two_dimentions=self.visualization_utility,
                decomposition_args=self.decomposition_args,
            )
            one_dimension = np.array([[i] for i in projection[:, 0]])

            model = KMeans(n_clusters=2, n_init="auto", random_state=self.random_seed)
            model.fit(one_dimension)
            labels = model.predict(one_dimension)
            centers = model.cluster_centers_

            # Labels for the split selection
            label_zero = np.where(labels == 0)
            label_one = np.where(labels == 1)

            # The indices of the
            left_child = indices[label_zero]
            right_child = indices[label_one]

            # Total scatter value calculation for the selection of the next
            # cluster to split.
            centered = util.center_data(self.X[indices, :])
            split_criterion = np.linalg.norm(centered, ord="fro")

            right_min = np.min(one_dimension[label_one])
            left_max = np.max(one_dimension[label_zero])
            if left_max > right_min:
                right_min = np.min(one_dimension[label_zero])
                left_max = np.max(one_dimension[label_one])

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
            "left_indices": left_child,
            "right_indices": right_child,
            "projection": projection,
            "centers": centers,
            "splitpoint": splitpoint,
            "split_criterion": split_criterion,
            "split_permission": flag,
            "color_key": key,
            "dendrogram_check": False,
        }

    @property
    def decomposition_method(self):
        return self._decomposition_method

    @decomposition_method.setter
    def decomposition_method(self, v):
        if not (v in ["pca", "kpca", "ica", "tsne"]):
            raise ValueError(
                "KMPDDP: decomposition_method: "
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
            raise ValueError(
                "KMPDDP: min_sample_split: Invalid value it should be int and > 1"
            )
        self._max_clusters_number = v

    @property
    def min_sample_split(self):
        return self._min_sample_split

    @min_sample_split.setter
    def min_sample_split(self, v):
        if v < 0 or (not isinstance(v, int)):
            raise ValueError(
                "KMPDDP: min_sample_split: Invalid value it should be int and > 1"
            )
        self._min_sample_split = v

    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, v):
        if v is not None and (not isinstance(v, int)):
            raise ValueError(
                "KMPDDP: min_sample_split: Invalid value it should be int and > 1"
            )
        self._random_seed = v

    @property
    def visualization_utility(self):
        return self._visualization_utility

    @visualization_utility.setter
    def visualization_utility(self, v):
        if v is not True and v is not False:
            raise ValueError(
                "KMPDDP: visualization_utility: Should be True or False"
            )

        if v is True and self.decomposition_method not in ["pca", "ica", "kpca"]:
            raise ValueError(
                "KMPDDP: visualization_utility: Should be pca when visualization_utility is True"
            )
        self._visualization_utility = v

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, v):
        self._tree = v

    @property
    def output_matrix(self):
        nds = self.tree.nodes
        output_matrix = [np.zeros(self.samples_number)]

        # The dictionary of nodes contains the created node from the KMPDDP
        # algorithm sorted from the root to the last split.
        for j in nds:
            # For the output matrix we don't want the leaves of the tree. Each
            # level of the output matrix represents a split the split exist in
            # the internal nodes of the tree. Only by checking the children of
            # those nodes we can extract the data for the current split.
            if not nds[j].is_leaf():
                # create output cluster splitting matrix
                tmp = np.copy(output_matrix[-1])
                # Left child according to the tree creation process
                tmp[self.tree.children(j)[0].data["indices"]] =\
                    self.tree.children(j)[0].identifier
                # Right child according to the tree creation process
                tmp[self.tree.children(j)[1].data["indices"]] =\
                    self.tree.children(j)[1].identifier

                # The output_matrix is created transposed
                output_matrix.append(tmp)
        # the first row contains only zeros
        del output_matrix[0]

        # transpose the output_matrix to be extracted
        output_matrix = np.array(output_matrix).transpose()

        return output_matrix

    @output_matrix.setter
    def output_matrix(self, v):
        raise RuntimeError(
            "KMPDDP: output_matrix: can only be generated and not to be assigned!"
        )

    @property
    def labels_(self):
        labels_ = np.ones(np.size(self.X, 0))
        for i in self.tree.leaves():
            labels_[i.data["indices"]] = i.identifier
        return labels_

    @labels_.setter
    def labels_(self, v):
        raise RuntimeError(
            "KMPDDP: labels_: can only be generated and not to be assigned!"
        )


class PDDP:
    """
    Class PDDP. It executes the PDDP algorithm.

    References
    ----------
    Boley, D. (1998). Principal direction divisive partitioning. Data mining
    and knowledge discovery, 2(4), 325-344.

    Parameters
    ----------
    decomposition_method : str, (optional)
        One of the ('pca', 'kpca', 'ica', 'tsne') supported decomposition
        methods used as kernel for the PDDP algorithm.
    max_clusters_number : int, (optional)
        Desired maximum number of clusters for the algorithm.
    min_sample_split : int, (optional)
        The minimum number of points needed in a cluster for a split to occur.
    visualization_utility : bool, (optional)
        If (True) generate the data needed by the visualization utilities of
        the package otherwise, if false the split_visualization and
        interactive_visualization of the package can not be created. For the
        'tsne' decomposition method does not support visualization because it
        affects the correct execution of the PDDP algorithm.
    **decomposition_args :
        Arguments for each of the decomposition methods ("decomposition.PCA" as
        "pca", "decomposition.KernelPCA" as "kpca", "decomposition.FastICA" as
        "ica", "manifold.TSNE" as "tsne") utilized by the HiPart package, as
        documented in the scikit-learn package, from which they are implemented.

    Attributes
    ----------
    output_matrix : numpy.ndarray
        Model's step by step execution output.
    labels_ :
        Extracted clusters from the algorithm.
    tree : treelib.Tree
        The object which contains all the information about the execution of
        the iPDDP algorithm.
    samples_number : int
        The number of samples contained in the data.

    """

    def __init__(
        self,
        decomposition_method="pca",
        max_clusters_number=100,
        min_sample_split=5,
        visualization_utility=True,
        **decomposition_args,
    ):
        self.decomposition_method = decomposition_method
        self.max_clusters_number = max_clusters_number
        self.min_sample_split = min_sample_split
        if decomposition_method in ["tsne"]:
            self.visualization_utility = False
            warnings.warn("PDDP: does not support visualization for 'tsne'.")
        else:
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
        indices = np.arange(X.shape[0])

        # initialize tree and root node                         # step (0)
        tree = Tree()
        # nodes unique IDs indicator
        self.node_ids = 0
        # nodes next color indicator (necessary for visualization purposes)
        self.cluster_color = 0
        tree.create_node(
            tag="cl_" + str(self.node_ids),
            identifier=self.node_ids,
            data=self.calculate_node_data(indices, self.cluster_color),
        )
        # indicator for the next node to split
        selected = 0

        # if no possibility of split exists on the data     # (ST2)
        if not tree.get_node(0).data["split_permission"]:
            raise RuntimeError("PDDP: cannot split the data at all!!!")

        # Initialize the ST1 stopping criterion counter that count the number
        # of clusters                                       # (ST1)
        counter = 1
        while (selected is not None) and (
            counter < self.max_clusters_number
        ):  # (ST1) or (ST2)

            self.split_function(tree, selected)  # step (1)

            # select the next kid for split based on the local minimum density
            selected = self.select_kid(tree.leaves())  # step (2)
            counter = counter + 1  # (ST1)

        self.tree = tree
        return self

    def fit_predict(self, X):
        """
        Execute the PDDP algorithm and return the results of the execution in
        the form of labels.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix with the samples on the rows and the variables on the
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

        Because python passes by reference data this function doesn't need a
        return statement.

        Parameters
        ----------
        tree : treelib.tree.Tree
            The tree build by the PDDP algorithm, in order to cluster the
            input data.
        selected_node : int
            The numerical identifier for the tree node that i about to be split.

        Returns
        -------
            There no returns in this function. The results of this function
            pass to execution by utilizing the python's pass-by-reference
            nature.

        """

        node = tree.get_node(selected_node)
        node.data["split_permission"] = False

        # left child indices extracted from the nodes split-point and the
        # indices included in the parent node
        left_kid_index = node.data["indices"][
            np.where(node.data["projection"][:, 0] < node.data["splitpoint"])[0]
        ]
        # right child indices
        right_kid_index = node.data["indices"][
            np.where(node.data["projection"][:, 0] >= node.data["splitpoint"])[0]
        ]

        # Nodes and data creation for the children
        # Uses the calculate_node_data function to create the data for the node
        tree.create_node(
            tag="cl" + str(self.node_ids + 1),
            identifier=self.node_ids + 1,
            parent=node.identifier,
            data=self.calculate_node_data(
                left_kid_index,
                node.data["color_key"],
            ),
        )
        tree.create_node(
            tag="cl" + str(self.node_ids + 2),
            identifier=self.node_ids + 2,
            parent=node.identifier,
            data=self.calculate_node_data(
                right_kid_index,
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

        This function creates the necessary cause for the stopping criterion
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

        maximum_scatter = None

        # Remove the nodes that can not split further
        scatter_leaves = list(
            np.array(leaves)[
                [
                    True if i.data["split_criterion"] is not None else False
                    for i in leaves
                ]
            ]
        )

        if len(scatter_leaves) > 0:
            for i in sorted(
                enumerate(scatter_leaves),
                key=lambda x: x[1].data["split_criterion"],
                reverse=True,
            ):
                if i[1].data["split_permission"]:
                    maximum_scatter = i[1].identifier
                    break

        return maximum_scatter

    def calculate_node_data(self, indices, key):
        """
        Calculation of the projections onto the Principal Components with the
        utilization of the "Principal Components Analysis" or the "Kernel
        Principal Components Analysis" or the "Independent Component Analysis"
        or "t-SNE" methods.

        The projection's clusters are split on the median pf the projected
        data.

        This function leads to the second Stopping criterion 2 of the
        algorithm.

        Parameters
        ----------
        indices : numpy.ndarray
            The index of the samples in the original data matrix.
        key : int
            The value of the color for each node.

        Returns
        -------
        data : dictionary
            The necessary data for each node which are splitting point.

        """

        projection = None
        splitpoint = None
        split_criterion = None
        flag = False

        # if the number of samples
        if indices.shape[0] > self.min_sample_split:

            centered = util.center_data(self.X[indices, :])

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

        return {
            "indices": indices,
            "projection": projection,
            "splitpoint": splitpoint,
            "split_criterion": split_criterion,
            "split_permission": flag,
            "color_key": key,
            "dendrogram_check": False,
        }

    @property
    def decomposition_method(self):
        return self._decomposition_method

    @decomposition_method.setter
    def decomposition_method(self, v):
        if not (v in ["pca", "kpca", "ica", "tsne"]):
            raise ValueError(
                "PDDP: decomposition_method: " + str(v) + ": Unknown decomposition method!"
            )
        self._decomposition_method = v

    @property
    def max_clusters_number(self):
        return self._max_clusters_number

    @max_clusters_number.setter
    def max_clusters_number(self, v):
        if v < 0 or (not isinstance(v, int)):
            raise ValueError(
                "PDDP: min_sample_split: Invalid value it should be int and > 1"
            )
        self._max_clusters_number = v

    @property
    def min_sample_split(self):
        return self._min_sample_split

    @min_sample_split.setter
    def min_sample_split(self, v):
        if v < 0 or (not isinstance(v, int)):
            raise ValueError(
                "PDDP: min_sample_split: Invalid value it should be int and > 1"
            )
        self._min_sample_split = v

    @property
    def visualization_utility(self):
        return self._visualization_utility

    @visualization_utility.setter
    def visualization_utility(self, v):
        if v is not True and v is not False:
            raise ValueError(
                "PDDP: visualization_utility: Should be True or False"
            )

        if v is True and self.decomposition_method not in ["pca", "ica", "kpca"]:
            raise ValueError(
                "PDDP: visualization_utility: Should be pca when visualization_utility is True"
            )
        self._visualization_utility = v

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, v):
        self._tree = v

    @property
    def output_matrix(self):
        nd_dict = self.tree.nodes
        output_matrix = [np.full(self.samples_number, 0)]

        # The dictionary of nodes contains the created node from the KMPDDP
        # algorithm sorted from the root to the last split.
        for k in nd_dict:
            # For the output matrix we don't want the leaves of the tree. Each
            # level of the output matrix represents a split the split exist in
            # the internal nodes of the tree. Only by checking the children of
            # those nodes we can extract the data for the current split.
            if not nd_dict[k].is_leaf():
                # create output cluster splitting matrix
                tmp = np.copy(output_matrix[-1])
                # Left child according to the tree creation process
                tmp[self.tree.children(k)[0].data["indices"]] = \
                    self.tree.children(k)[0].identifier
                # Right child according to the tree creation process
                tmp[self.tree.children(k)[1].data["indices"]] = \
                    self.tree.children(k)[1].identifier

                # The output_matrix is created transposed
                output_matrix.append(tmp)
        # the first row contains only zeros
        del output_matrix[0]

        # transpose the output_matrix to be extracted
        output_matrix = np.array(output_matrix).transpose()

        return output_matrix

    @output_matrix.setter
    def output_matrix(self, v):
        raise RuntimeError(
            "PDDP: output_matrix: can only be generated and not to be assigned!"
        )

    @property
    def labels_(self):
        labels_ = np.ones(np.size(self.X, 0))
        for i in self.tree.leaves():
            labels_[i.data["indices"]] = i.identifier
        return labels_

    @labels_.setter
    def labels_(self, v):
        raise RuntimeError(
            "PDDP: labels_: can only be generated and not to be assigned!"
        )


class BisectingKmeans:
    """
    Class BisectingKmeans. It executes the bisecting k-Means algorithm.

    References
    ----------
    Savaresi, S. M., & Boley, D. L. (2001, April). On the performance of
    bisecting K-means and PDDP. In Proceedings of the 2001 SIAM International
    Conference on Data Mining (pp. 1-14). Society for Industrial and Applied
    Mathematics.

    Parameters
    ----------
    max_clusters_number : int, (optional)
        Desired maximum number of clusters for the algorithm.
    min_sample_split : int, (optional)
        The minimum number of points needed in a cluster for a split to occur.
    random_seed : int, (optional)
        The random seed fed in the k-Means algorithm.

    Attributes
    ----------
    output_matrix : numpy.ndarray
        Model's step by step execution output.
    labels_ : numpy.ndarray
        Extracted clusters from the algorithm.
    tree : treelib.Tree
        The object which contains all the information about the execution of
        the bisecting k-Means algorithm.
    samples_number : int
        The number of samples contained in the data.

    """

    def __init__(self, max_clusters_number=100, min_sample_split=5, random_seed=None):
        self.max_clusters_number = max_clusters_number
        self.min_sample_split = min_sample_split
        self.random_seed = random_seed

    def fit(self, X):
        """
        Execute the BisectingKmeans algorithm and return all the execution
        data in the form of a BisectingKmeans class object.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix with the samples on the rows and the variables on the
            columns.

        Returns
        -------
        self
            A BisectingKmeans class type object, with complete results on the
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
        # indicator for the next node to split
        selected_node = 0

        # if no possibility of split exists on the data
        if not tree.get_node(0).data["split_permission"]:
            print("cannot split at all")
            return self

        # Initialize the ST1 stopping criterion counter that count the number
        # of clusters.
        found_clusters = 1
        while (selected_node is not None) and (
            found_clusters < self.max_clusters_number
        ):
            self.split_function(tree, selected_node)  # step (1)

            # select the next kid for split based on the local minimum density
            selected_node = self.select_kid(tree.leaves())  # step (2)
            found_clusters = found_clusters + 1  # (ST1)

        self.tree = tree
        return self

    def fit_predict(self, X):
        """
        Execute the BisectingKmeans algorithm and return the results of the
        execution in the form of labels.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix with the samples on the rows and the variables on the
            columns.

        Returns
        -------
        labels_ : numpy.ndarray
            Extracted clusters from the algorithm.

        """

        return self.fit(X).labels_

    def split_function(self, tree, selected_node):
        """
        Split the indicated node by clustering the data with a binary k-means
        clustering algorithm.

        Because python passes by reference data this function doesn't need a
        return statement.

        Parameters
        ----------
        tree : treelib.tree.Tree
            The tree build by the BisectingKmeans algorithm, in order to
            cluster the input data.
        selected_node : int
            The numerical identifier for the tree node that i about to be split.

        Returns
        -------
            There is no returns in this function. The results of this function
            pass to execution by utilizing the python's pass-by-reference
            nature.

        """
        node = tree.get_node(selected_node)
        node.data["split_permission"] = False

        # left child indices extracted from the nodes split-point and the
        # indices included in the parent node
        left_index = node.data["left_indices"]

        # right child indices
        right_index = node.data["right_indices"]

        # Nodes and data creation for the children
        # Uses the calculate_node_data function to create the data for the node
        tree.create_node(
            tag="cl" + str(self.node_ids + 1),
            identifier=self.node_ids + 1,
            parent=node.identifier,
            data=self.calculate_node_data(
                left_index,
                self.X[left_index, :],
                node.data["color_key"],
            ),
        )
        tree.create_node(
            tag="cl" + str(self.node_ids + 2),
            identifier=self.node_ids + 2,
            parent=node.identifier,
            data=self.calculate_node_data(
                right_index,
                self.X[right_index, :],
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

        This function creates the necessary data for further execution of the
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
                if i[1].data["split_permission"]:
                    next_split = i[1].identifier
                    break

        return next_split

    def calculate_node_data(self, indices, data_matrix, key):
        """
        Execution of the binary k-Means algorithm on the samples presented by
        the data_matrix. The two resulted clusters are the two new clusters if
        the leaf is chosen to be split. And calculation of the splitting
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
            The necessary data for each node which are splitting point.

        """
        # if the number of samples
        if indices.shape[0] > self.min_sample_split:

            model = KMeans(n_clusters=2, n_init="auto", random_state=self.random_seed)
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
            "left_indices": left_child,
            "right_indices": right_child,
            "centers": centers,
            "split_criterion": split_criterion,
            "split_permission": flag,
            "color_key": key,
            "dendrogram_check": False,
        }

    @property
    def max_clusters_number(self):
        return self._max_clusters_number

    @max_clusters_number.setter
    def max_clusters_number(self, v):
        if v < 0 or (not isinstance(v, int)):
            raise ValueError(
                "BisectingKmeans: min_sample_split: Invalid value it should be int and > 1"
            )
        self._max_clusters_number = v

    @property
    def min_sample_split(self):
        return self._min_sample_split

    @min_sample_split.setter
    def min_sample_split(self, v):
        if v < 0 or (not isinstance(v, int)):
            raise ValueError(
                "BisectingKmeans: min_sample_split: Invalid value it should be int and > 1"
            )
        self._min_sample_split = v

    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, v):
        if v is not None and (not isinstance(v, int)):
            raise ValueError(
                "BisectingKmeans: min_sample_split: Invalid value it should be int and > 1"
            )
        self._random_seed = v

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, v):
        self._tree = v

    @property
    def output_matrix(self):
        tnds = self.tree.nodes
        output_matrix = [np.full(self.samples_number, 0)]

        # The dictionary of nodes contains the created node from the KMPDDP
        # algorithm sorted from the root to the last split.
        for t in tnds:
            # For the output matrix we don't want the leaves of the tree. Each
            # level of the output matrix represents a split the split exist in
            # the internal nodes of the tree. Only by checking the children of
            # those nodes we can extract the data for the current split.
            if not tnds[t].is_leaf():
                # create output cluster splitting matrix
                tmp = np.copy(output_matrix[-1])
                # Left child according to the tree creation process
                tmp[self.tree.children(t)[0].data["indices"]] = \
                    self.tree.children(t)[0].identifier
                # Right child according to the tree creation process
                tmp[self.tree.children(t)[1].data["indices"]] = \
                    self.tree.children(t)[1].identifier

                # The output_matrix is created transposed
                output_matrix.append(tmp)
        # the first row contains only zeros
        del output_matrix[0]

        # transpose the output_matrix to be extracted
        output_matrix = np.array(output_matrix).transpose()

        return output_matrix

    @output_matrix.setter
    def output_matrix(self, v):
        raise RuntimeError(
            "BisectingKmeans: output_matrix: can only be generated and not to be assigned!"
        )

    @property
    def labels_(self):
        labels_ = np.ones(np.size(self.X, 0))
        for i in self.tree.leaves():
            labels_[i.data["indices"]] = i.identifier
        return labels_

    @labels_.setter
    def labels_(self, v):
        raise RuntimeError(
            "BisectingKmeans: labels_: can only be generated and not to be assigned!"
        )


class MDH:
    """
    Class MDH. It executes the MDH algorithm.

    References
    ----------
    Pavlidis, N. G., Hofmeyr, D. P., & Tasoulis, S. K. (2016). Minimum density
    hyperplanes. Journal of Machine Learning Research, 17 (156), 1-33.

    Parameters
    ----------
    max_clusters_number : int, optional
        Desired maximum number of clusters to find the MDH algorithm.
    max_iterations : int, optional
        Maximum number of iterations on the search for the minimum density
        hyperplane.
    k : float, optional
        The multiples of the standard deviation which the existence of a
        splitting hyperplane is allowed. The default value is 2.3.
    percentile : float, optional
        The percentile distance from the dataset's edge in which a split can
        not occur. [0,0.5) values are allowed.
    min_sample_split : int, optional
        The minimum number of points needed in a cluster for a split to occur.
    random_state : int, optional
        The random seed to be used in the algorithm's execution.

    Attributes
    ----------
    output_matrix : numpy.ndarray
        Model's step by step execution output.
    labels_ : numpy.ndarray
        Extracted clusters from the algorithm.
    tree : treelib.Tree
        The object which contains all the information about the execution of
        the MDH algorithm.
    samples_number : int
        The number of samples contained in the data.

    """

    def __init__(
        self,
        max_clusters_number=100,
        max_iterations=10,
        k=2.3,
        percentile=0.1,
        min_sample_split=5,
        random_state=None,
    ):
        self.max_clusters_number = max_clusters_number
        self.k = k
        self.max_iterations = max_iterations
        self.percentile = percentile
        self.min_sample_split = min_sample_split
        self.random_state = random_state

    def fit(self, X):
        """
        Execute the MDH algorithm and return all the execution data in the form
        of a MDH class object.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix with the samples on the rows and the variables on the
            columns.

        Returns
        -------
        self
            A MDH class type object, with complete results on the algorithm's
            analysis.

        """

        # initialize the random seed
        np.random.seed(self.random_state)

        # initialize the data matrix and the number of samples
        self.X = X
        self.samples_number = np.size(X, 0)

        # create an id vector for the samples of X
        indices = np.array([int(i) for i in range(self.samples_number)])

        # initialize the tree and root node                           # step (0)
        den_tree = Tree()
        # nodes unique IDs indicator
        self.node_ids = 0
        # nodes next color indicator (necessary for visualization purposes)
        self.cluster_color = 0
        den_tree.create_node(
            tag="density_cluster_" + str(self.node_ids),
            identifier=self.node_ids,
            data=self.calculate_node_data(indices, self.cluster_color),
        )
        # indicator for the next node to split
        selected_node = 0

        # if no possibility of split exists on the data                  # (ST2)
        if not den_tree.get_node(0).data["split_permission"]:
            raise RuntimeError("MDH: cannot split the data at all!!!")

        # Initialize the ST1 stopping criterion counter that count the number
        # of clusters                                                    # (ST1)
        found_clusters = 1
        while (found_clusters < self.max_clusters_number) and (
            selected_node is not None
        ):  # (ST1) or (ST2)

            self.split_function(den_tree, selected_node)  # step (1, 2)

            # select the next kid for split based on the local minimum density
            selected_node = self.select_kid(den_tree.leaves())  # step (3)
            found_clusters = found_clusters + 1  # (ST1)

        self.tree = den_tree
        return self

    def fit_predict(self, X):
        """
        Execute the MDH algorithm and return the results of the execution in the
        form of labels.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix with the samples on the rows and the variables on the
            columns.

        Returns
        -------
        labels_ : numpy.ndarray
            Extracted clusters from the algorithm.

        """

        return self.fit(X).labels_

    def split_function(self, den_tree, selected_node):
        """
        Split the indicated node on the minimum of the local minimum density
        of the data projected on the first principal component.

        Because python passes by reference data this function doesn't need a
        return statement.

        Parameters
        ----------
        den_tree : treelib.tree.Tree
            The tree build by the MDH algorithm, in order to cluster the
            input data.
        selected_node : int
            The numerical identifier for the tree node that i about to be split.

        Returns
        -------
            There no returns in this function. The results of this function
            pass to execution by utilizing the python's pass-by-reference
            nature.

        """
        node = den_tree.get_node(selected_node)
        node.data["split_permission"] = False

        # left child indices extracted from the nodes split-point and the
        # indices included in the parent node
        left_kid_index = node.data["indices"][
            np.where(node.data["projection"][:, 0] < node.data["splitpoint"])[0]
        ]
        # right child indices
        right_kid_index = node.data["indices"][
            np.where(node.data["projection"][:, 0] >= node.data["splitpoint"])[0]
        ]

        # Nodes and data creation for the children
        # Uses the calculate_node_data function to create the data for the node
        den_tree.create_node(
            tag="density_cluster_" + str(self.node_ids + 1),
            identifier=self.node_ids + 1,
            parent=node.identifier,
            data=self.calculate_node_data(left_kid_index, node.data["color_key"]),
        )
        den_tree.create_node(
            tag="density_cluster_" + str(self.node_ids + 2),
            identifier=self.node_ids + 2,
            parent=node.identifier,
            data=self.calculate_node_data(right_kid_index, self.cluster_color + 1),
        )

        self.cluster_color += 1
        self.node_ids += 2

    def select_kid(self, possible_splits):
        """
        The clusters each time exist in the leaves of the trees. From those
        leaves select the next leave to split based on the algorithm's
        specifications.

        This function creates the necessary cause for the stopping criterion
        ST1.

        Parameters
        ----------
        possible_splits : list of treelib.node.Node
            The list of nodes needed to exam to select the next Node to split.

        Returns
        -------
        next_split : int
            The identifier of the next node to split by the algorithm.

        """
        min_density_node = None

        # Remove the nodes that can not split further
        possible_splits = list(
            np.array(possible_splits)[
                [
                    True if i.data["split_criterion"] is not None else False
                    for i in possible_splits
                ]
            ]
        )

        if len(possible_splits) > 0:
            for i in sorted(
                enumerate(possible_splits), key=lambda x: x[1].data["split_criterion"],
                reverse=True,
            ):
                if i[1].data["split_permission"]:
                    min_density_node = i[1].identifier
                    break

        return min_density_node

    def calculate_node_data(self, indices, key):
        """
        Find a minimum density hyperplane to bisect the data. The determination
        of the minimum density hyperplane is based on the minimization is found
        by minimizing the first derivative of the density function. This is made
        possible through the use of "Sequential Quadratic Programming" (SQP)
        method, which is used to simultaneously find the optimal projection
        vector v and minimum density point b.

        This function leads to the second Stopping criterion 2 of the
        algorithm.

        Parameters
        ----------
        indices : numpy.ndarray
            The index of the samples in the original data matrix.
        key : int
            The value of the color for each node.

        Returns
        -------
        data : dict
            The necessary data for each node which are splitting point.

        """

        # Initialization of the return variables
        projection = None
        splitpoint = None
        split_criterion = None
        flag = False
        split_vector = None

        # If the number of samples in the node is greater than the minimum allowed for split
        if indices.shape[0] > self.min_sample_split:
            node_data = self.X[indices, :]
            node_size = node_data.shape[0]
            # Normalize the data of the node to zero mean and unit standard deviation
            node_data = (node_data - np.mean(node_data, 0)) / np.std(node_data, 0)

            minC = 100 if node_size * self.percentile > 100 else node_size * self.percentile

            solutions = []
            for i in range(0, self.max_iterations):

                # Generate a random vector in the space of the node's data and
                # normalize it to unit length
                # v_n_b: vector v and point b
                initial_v_n_b = stats.norm.rvs(size=np.shape(node_data)[1])
                initial_v_n_b = initial_v_n_b / np.linalg.norm(initial_v_n_b)
                initial_v_n_b = np.append(initial_v_n_b, 0)

                # Find the minimum density point of the data on the projection
                # direction
                minimum_b = util.initialize_b(initial_v_n_b, node_data, depth_init=True)

                if minimum_b:
                    initial_v_n_b[-1] = minimum_b
                    # res has the following fields that are of interest:
                    #   1. success (whether algorithm terminated successfully)
                    #   2. x (solution)
                    #   3. nfev (number of function evaluations)
                    #   4. njev (number of jacobian/ gradient evaluations)
                    results, depth = util.md_sqp(initial_v_n_b, node_data, self.k)

                    # If the algorithm terminated successfully try to append the solution
                    if results.success:
                        v = results.x[:-1] / np.linalg.norm(results.x[:-1])
                        projection = np.dot(node_data, v).reshape(-1, 1)
                        b = results.x[-1]
                        c0 = np.sum(projection > b)
                        # Solutions in the edges of the projection are not acceptable
                        if min(c0, node_size - c0) >= minC:
                            solutions.append((v, b, depth))

            # Find the solution with the minimum depth
            if solutions:
                split = min(solutions, key=lambda x: x[2])
                if split:
                    splitpoint = split[1]
                    projection = np.dot(node_data, split[0]).reshape(-1, 1)
                    split_criterion = indices.shape[0]
                    flag = True
                    split_vector = split[0]

        return {
            "indices": indices,
            "projection": projection,
            "splitpoint": splitpoint,
            "split_criterion": split_criterion,
            "split_permission": flag,
            "split_vector": split_vector,
            "color_key": key,
            "dendrogram_check": False,
        }

    @property
    def max_clusters_number(self):
        return self._max_clusters_number

    @max_clusters_number.setter
    def max_clusters_number(self, v):
        if v < 0 or (not isinstance(v, int)):
            raise ValueError(
                "MDH: min_sample_split: Invalid value it should be int and > 1"
            )
        self._max_clusters_number = v

    @property
    def max_iterations(self):
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, v):
        if v < 0 or (not isinstance(v, int)):
            raise ValueError(
                "MDH: max_iteration: Invalid value it should be int and > 1"
            )
        self._max_iterations = v

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, v):
        if v < 0 or (not isinstance(v, float)):
            raise ValueError(
                "MDH: k: Invalid value it should be float and > 1"
            )
        self._k = v

    @property
    def percentile(self):
        return self._percentile

    @percentile.setter
    def percentile(self, v):
        if v >= 0.5 or v < 0:
            raise ValueError("MDH: percentile: Should be between [0,0.5) interval")
        self._percentile = v

    @property
    def min_sample_split(self):
        return self._min_sample_split

    @min_sample_split.setter
    def min_sample_split(self, v):
        if v < 0 or (not isinstance(v, int)):
            raise ValueError(
                "MDH: min_sample_split: Invalid value it should be int and > 1"
            )
        self._min_sample_split = v

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, v):
        if v is not None and (not isinstance(v, int)):
            raise ValueError(
                "MDH: min_sample_split: Invalid value it should be int and > 1"
            )
        self._random_state = v

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, v):
        self._tree = v

    @property
    def output_matrix(self):
        nd_dict = self.tree.nodes
        output_matrix = [np.zeros(np.size(self.X, 0))]

        # the dictionary of nodes contains the created node from the MDH
        # algorithm sorted from the root to the last split
        for i in nd_dict:
            # For the output matrix we don't want the leaves of the tree. Each
            # level of the output matrix represents a split the split exist in
            # the internal nodes of the tree. Only by checking the children of
            # those nodes we can extract the data for the current split.
            if not nd_dict[i].is_leaf():
                # create output cluster splitting matrix
                tmp = np.copy(output_matrix[-1])
                # Left child according to the tree creation process
                tmp[self.tree.children(i)[0].data["indices"]] = self.tree.children(i)[
                    0
                ].identifier
                # Right child according to the tree creation process
                tmp[self.tree.children(i)[1].data["indices"]] = self.tree.children(i)[
                    1
                ].identifier

                # The output_matrix is created transposed
                output_matrix.append(tmp)
        # the first row contains only zeros
        del output_matrix[0]

        # transpose the output_matrix to be extracted
        output_matrix = np.array(output_matrix).transpose()

        return output_matrix

    @output_matrix.setter
    def output_matrix(self, v):
        raise RuntimeError(
            "MDH: output_matrix: can only be generated and not to be assigned!"
        )

    @property
    def labels_(self):
        labels_ = np.ones(np.size(self.X, 0))
        for i in self.tree.leaves():
            labels_[i.data["indices"]] = i.identifier
        return labels_

    @labels_.setter
    def labels_(self, v):
        raise RuntimeError(
            "MDH: labels_: can only be generated and not to be assigned!"
        )
