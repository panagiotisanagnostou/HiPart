# -*- coding: utf-8 -*-
"""
Application of the Principal Direction Divisive Partitioning (PDDP) algorithm.

@author: Panagiotis Anagnostou
"""

import HIDIV.__utility_functions as util
import numpy as np

from treelib import Tree

class PDDP:
    """
    Class PDDP. It executes the PDDP algorithm.
    
    Parameters
    ----------
    decomposition_method : str, optional
        One of the supported dimentionality reduction methods used as kernel for the PDDP algorithm.
    max_clusters_number : int, optional
        Desired maximum number of clusters for the algorithm.
    min_sample_split : int, optional
        Minimum number of points each cluster should contain selected by the user.
    **decomposition_args : 
        arguments for each of the decomposition methods utilized by the HIDIV package 
        
    output_matrix : numpy.ndarray
        Model's step by step execution output.
    labels_ :
        Extracted clusters from the algorithm
        
    Methodes
    --------
    fit(X)
        Execute the PDDP algorithm and return all the execution data in the form of a PDDP class object.
        
    fit_predict(X)
        Execute the PDDP algorithm and return the results of the execution in the form of labels.
        
    split_fuction(tree, selected_node):
        Split the indicated node on the minimum of the local minimum density of the data projected on the first principal component.

        Because python passes by refference data this function doesn't need a return statment.
        
    select_kid(leaves)
        The clusters each time exist in the leaves of the trees. From those leaves select the next leave to split based on the algorithm's specifications.

        This function creates the nescesary cause for the stopping criterion ST1.
        
    calculate_node_data(indices, data_matrix, key)
        Calculation of the projections on to the first 2 Components with the utilization of the "Principal Components Analysis", "Kernel Principal Components Analysis" or "Independent Component Analysis" methods. 

        Determination of the projection's clusters by utilizing the binary k-means clustering algorithm.

        This function leads to the second Stopping criterion 2 of the algorithm.
    
    """ 
    def __init__(self, decomposition_method='pca', max_clusters_number = 100, min_sample_split = 5, **decomposition_args):
        self.decomposition_method = decomposition_method
        self.max_clusters_number = max_clusters_number
        self.min_sample_split = min_sample_split
        self.decomposition_args = decomposition_args
    
    
    #%% Main algorithm execution methods
    def fit(self, X):
        """
        Execute the PDDP algorithm and return all the execution data in the form of a PDDP class object.
        
        Parameters
        ----------
        X: numpy.ndarray
            data matrix (must check and return an error if not)
            
        Returns
        -------
        self
            A PDDP class type object, with complete results on the algorithm's analysis
        
        """
        self.X = X
        self.samples_number = X.shape[0]
        
        # create an id vector for the samples of X
        indices = np.array([ int(i) for i in range(np.size(self.X, 0)) ])
        
        # initialize tree and root node                         # step (0)
        tree = Tree()
        # nodes unique IDs indicator
        self.node_ids = 0
        # nodes next color indicator (necessary for visualization purposes)
        self.cluster_color = 0
        tree.create_node(tag = 'cl_' + str(self.node_ids), identifier = self.node_ids, data = self.calculate_node_data(indices, self.X, self.cluster_color))
        # inidcator for the next node to split
        selected_node = 0
        
        # if no possibility of split exists on the data     # (ST2)
        if not tree.get_node(0).data['split_permition']:                  
            print("cannot split at all")
            return self
        
        # Initialize the ST1 stopping critirion counter that count the number 
        # of clusters                                       # (ST1)
        found_clusters = 1
        while selected_node != None and found_clusters < self.max_clusters_number:  # (ST1) or (ST2)
        
            self.split_function(tree, selected_node)            # step (1)
            
            # select the next kid for split based on the local minimum density
            selected_node = self.select_kid(tree.leaves())      # step (2)
            found_clusters = found_clusters +1              # (ST1)
        
        
        self.tree = tree 
        return self
    
    
    def fit_predict(self, X):
        """
        Execute the PDDP algorithm and return the results of the execution in the form of labels.
        
        Parameters
        ----------
        X : numpy.ndarray
            data matrix (numpy array, must check and return an error if not)
        
        Returns
        -------
        labels_ : numpy.ndarray
            The execution labels extracted labels from the PDDP algorithm.
        
        """
        
        return self.fit(X).labels_
    
    
    def split_function(self, tree, selected_node):
        """
        Split the indicated node on the minimum of the local minimum density of the data projected on the first principal component.
        
        Because python passes by refference data this function doesn't need a return statment.
        
        Parameters
        ----------
        tree : treelib.tree.Tree
            The tree build by the PDDP algorithm, in order to cluster the input data
        
        Returns
        -------
            There no returns in this function. The results of this funciton pass to execution by utilizing the python's pass-by-reference nature.
        
        """
        node = tree.get_node(selected_node)
        node.data['split_permition'] = False
        
        # left child indecies extracted from the nodes splitpoint and the 
        # indecies included in the parent node
        left_kid_index = node.data['indices'][ np.where(node.data['projection'][:,0] < node.data['splitpoint'])[0] ]
        # right child indecies
        right_kid_index = node.data['indices'][ np.where(node.data['projection'][:,0] >= node.data['splitpoint'])[0] ]
        
        # Nodes and data creation for the children
        # Uses the calculate_node_data function to create the data for the node
        tree.create_node(tag = 'cl' + str(self.node_ids + 1), identifier = self.node_ids + 1, parent = node.identifier, data = self.calculate_node_data(left_kid_index, self.X[left_kid_index,:], node.data['color_key']))
        tree.create_node(tag = 'cl' + str(self.node_ids + 2), identifier = self.node_ids + 2, parent = node.identifier, data = self.calculate_node_data(right_kid_index, self.X[right_kid_index,:], self.cluster_color+1))
        
        self.cluster_color += 1
        self.node_ids += 2
    
    
    def select_kid(self, leaves):
        """
        The clusters each time exist in the leaves of the trees. From those leaves select the next leave to split based on the algorithm's specifications.
        
        This function creates the nescesary cause for the stopping criterion ST1.
        
        Parameters
        ----------
        leaves : list of treelib.node.Node
            The list of nodes needed to exam to select the next Node to split
        
        Returns
        -------
        int
            A PDDP class type object, with complete results on the algorithm's analysis
            
        """
        minimum_location = None
        
        # Remove the nodes that can not split further
        leaves = list(np.array(leaves)[ [ True if not (i.data['split_criterion'] == None) else False for i in leaves ] ])
        
        if len(leaves) > 0:
            for i in sorted(enumerate(leaves), key=lambda x:x[1].data['split_criterion'], reverse=True):
                if i[1].data['split_permition']:
                    minimum_location = i[1].identifier
                    break
        
        return minimum_location
    
    
    def calculate_node_data(self, indices, data_matrix, key):
        """
        Calculation of the projections on to the first 2 Components with the utilization of the "Principal Components Analysis", "Kernel Principal Components Analysis" or "Independent Component Analysis" methods. 

        Determination of the projection's clusters by utilizing the binary k-means clustering algorithm.

        This function leads to the second Stopping criterion 2 of the algorithm. 
        
        Parameters
        ----------
        indices : numpy.ndarray
            The index of the samples in the original data matrix
            
        data_matrix : numpy.ndarray
            The data matrix containing all the data for the samples
        
        key : int
            The value of the color for each node
            
        Returns
        -------
        data : dictionary
            The necesary data for each node which are spliting point
            
        """
        # if the number of samples 
        if indices.shape[0] > self.min_sample_split:
            
            centered = util.center_data(data_matrix)
            
            # execute pca on the data matrix
            two_dimensions = util.execute_decomposition_method(centered, self.decomposition_method, self.decomposition_args)
            
            # Total scatter value calculation for the selection of the next cluster to split.
            scat = np.linalg.norm(centered, ord="fro")
            
            if not np.allclose(scat, 0):
                splitpoint = .0
                split_criterion = scat
                flag = True
            else:
                splitpoint = None       # (ST2)
                split_criterion = None  # (ST2)
                flag = False            # (ST2)
        # =========================
        else:
            two_dimensions = None
            splitpoint = None           # (ST2)
            split_criterion = None      # (ST2)
            flag = False                # (ST2)
        
        return {'indices': indices, 'projection': two_dimensions, 'splitpoint': splitpoint, 'split_criterion': split_criterion, 'split_permition': flag, 'color_key': key, "dendrogram_check": False}
    
    
   
    #%% Properties
    @property
    def decomposition_method(self):
        return self._decomposition_method
    
    @decomposition_method.setter
    def decomposition_method(self, v):
        if not (v in ['pca', 'kpca', 'ica']):
            raise ValueError("decomposition_method: " + str(v) + ": Unknown decomposition method!")
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
    def tree(self):
        return self._tree
    
    @tree.setter
    def tree(self, v):
        self._tree = v
        
    @property
    def output_matrix(self):
        ndDict = self.tree.nodes
        output_matrix = [ np.zeros(np.size(self.X, 0)) ]
        for i in ndDict:
            if not ndDict[i].is_leaf():
                # create output cluster spliting matrix
                tmp = np.copy(output_matrix[-1])
                tmp[ self.tree.children(i)[0].data['indices'] ] = self.tree.children(i)[0].identifier
                tmp[ self.tree.children(i)[1].data['indices'] ] = self.tree.children(i)[1].identifier
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
            labels_[i.data['indices']] = i.identifier
        self.labels_ = labels_
        return self._labels_
    
    @labels_.setter
    def labels_(self, v):
        self._labels_ = v