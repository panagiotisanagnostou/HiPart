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
"""

import numpy as np
import warnings


class Partition:
    """
    Class Partition. Is an abstract class that contains the necessary methods
    for the implementation of the Partition algorithms of the HiPart package.

    Parameters
    ----------
    decomposition_method : str, (optional)
        One of the ('pca', 'kpca', 'ica', 'tsne') supported decomposition
        methods used as kernel for the Partition algorithms it is implemented
        in the HiPart package.
    max_clusters_number : int, (optional)
        Desired maximum number of clusters to find the Partition algorithm.
    min_sample_split : int, (optional)
        The minimum number of points needed in a cluster for a split to occur.
    visualization_utility : bool, (optional)
        If (True) generate the data needed by the visualization utilities of
        the package otherwise, if false the split_visualization and
        interactive_visualization of the package can not be created. For the
        'tsne' decomposition method does not support visualization because it
        affects the correct execution of the dePDDP algorithm.
    distance_matrix : bool, (optional)
        If (True) the input data are considered as a distance matrix and not as
        a data matrix. The distance matrix is a square matrix with the samples
        on the rows and the variables on the columns. The distance matrix is
        used only in conjunction with the 'mds' decomposition method and no
        other from the supported decomposition methods.
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
        min_sample_split=5,
        visualization_utility=True,
        distance_matrix=False,
        **decomposition_args,
    ):
        self.decomposition_method = decomposition_method
        self.max_clusters_number = max_clusters_number
        self.min_sample_split = min_sample_split
        if decomposition_method in ["tsne"]:
            self.visualization_utility = False
            warnings.warn("does not support visualization for 'tsne'.")
        else:
            self.visualization_utility = visualization_utility
        self.distance_matrix = distance_matrix
        if self.distance_matrix:
            self.decomposition_method = "mds"
        self.decomposition_args = decomposition_args

    def fit(self, X):
        """
        Execution of the main workflow of the algorithm by clustering the input
        data, returning the results and creating the necessary data for the
        visualization utilities.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix with the samples on the rows and the variables on the
            columns. If the distance_matrix is True then X should be a square
            distance matrix.

        Returns
        -------
        self
            A dePDDP class type object, with complete results on the
            algorithm's analysis.

        """
        pass

    def fit_predict(self, X):
        """
        Returns the results of the fit method in the form of the labels of the
        predicted clustering labels.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix with the samples on the rows and the variables on the
            columns. If the distance_matrix is True then X should be a square
            distance matrix.

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

        Because python passes by reference data this function doesn't need a
        return statement.

        Parameters
        ----------
        tree : treelib.tree.Tree
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
            tag="cl_" + str(self.node_ids + 1),
            identifier=self.node_ids + 1,
            parent=node.identifier,
            data=self.calculate_node_data(left_kid_index, node.data["color_key"]),
        )
        tree.create_node(
            tag="cl_" + str(self.node_ids + 2),
            identifier=self.node_ids + 2,
            parent=node.identifier,
            data=self.calculate_node_data(right_kid_index, self.cluster_color + 1),
        )

        self.cluster_color += 1
        self.node_ids += 2

    def select_kid(self, possible_splits, decreasing=False):
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
        decreasing : bool, (optional)
            If True the function will select the node with the highest value.

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
                enumerate(possible_splits),
                key=lambda x: x[1].data["split_criterion"],
                reverse=decreasing,
            ):
                if i[1].data["split_permission"]:
                    min_density_node = i[1].identifier
                    break

        return min_density_node

    def calculate_node_data(self, indices, key):
        """
        Execution of the necessary calculations for the creation of the data for
        each node of the tree. The data are the following:

        1. The indices of the samples that are included in the node. ("indices")
        2. The projection of the samples on the first principal component.
        ("projection")
        3. The projection vectors of the samples on the first principal
        component. ("projection_vectors")
        3. The split point of the node. ("splitpoint")
        4. The split criterion of the node. ("split_criterion")
        5. The split permission of the node. ("split_permission")
        6. The color key of the node. ("color_key")
        7. The dendrogram check of the node. Utility variable for the dendrogram
        visualization. ("dendrogram_check")

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
        pass

    @property
    def decomposition_method(self):
        return self._decomposition_method

    @decomposition_method.setter
    def decomposition_method(self, v):
        if not (v in ["pca", "kpca", "ica", "tsne", "mds"]):
            raise ValueError(
                "decomposition_method: " + str(v) + ": Unknown decomposition method!"
            )
        self._decomposition_method = v

    @property
    def max_clusters_number(self):
        return self._max_clusters_number

    @max_clusters_number.setter
    def max_clusters_number(self, v):
        if v < 0 or (not isinstance(v, int)):
            raise ValueError("min_sample_split: Invalid value it should be int and > 1")
        self._max_clusters_number = v

    @property
    def min_sample_split(self):
        return self._min_sample_split

    @min_sample_split.setter
    def min_sample_split(self, v):
        if v < 0 or (not isinstance(v, int)):
            raise ValueError("min_sample_split: Invalid value it should be int and > 1")
        self._min_sample_split = v

    @property
    def visualization_utility(self):
        return self._visualization_utility

    @visualization_utility.setter
    def visualization_utility(self, v):
        if v is not True and v is not False:
            raise ValueError("visualization_utility: Should be True or False")

        if v is True and self.decomposition_method not in ["pca", "ica", "kpca", "mds"]:
            raise ValueError(
                "visualization_utility: "
                + str(self.decomposition_method)
                + ": is not supported from the HiPart package!"
            )
        self._visualization_utility = v

    @property
    def distance_matrix(self):
        return self._distance_matrix

    @distance_matrix.setter
    def distance_matrix(self, v):
        if v is not True and v is not False:
            raise ValueError("distance_matrix: Should be boolean (True or False)")
        self._distance_matrix = v

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
            "output_matrix: can only be generated and not to be assigned!"
        )

    @property
    def labels_(self):
        # Regenerate the labels_ each time from the tree leafs
        labels_ = np.ones(np.size(self.X, 0))
        # Iterate through the leaves of the tree and assign the labels in
        # increasing order based on the j counter variable
        j = 0
        for i in self.tree.leaves():
            labels_[i.data["indices"]] = j
            j += 1
        return labels_

    @labels_.setter
    def labels_(self, v):
        raise RuntimeError("labels_: can only be generated and not to be assigned!")
