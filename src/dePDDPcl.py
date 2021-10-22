# -*- coding: utf-8 -*-
"""
Application of the Principal Direction Divisive Partitioning (PDDP).

@author: Panagiotis Anagnostou
"""

import numpy as np
import statsmodels.api as sm

from KDEpy import FFTKDE
from scipy.signal import argrelextrema
from sklearn.decomposition import PCA, KernelPCA
from treelib import Tree

class dePDDP:
    """
    Class dePDDP. It executes the dePDDP algorithm 
    
    Parameters
    ----------
        reduction_method : str, optional
            One of the supported dimentionality reduction methods used as kernel for the dePDDP algorithm.
        
        max_clusters_number : int, optional
            Desired maximum number of clusters for the algorithm.
            
        bandwidth_scale : float, optional
            Standard deviation scaler for the density aproximation. Allowed values are in the (0,1).
            
        percentile : float, optional
            The peprcentile of the entirety of the dataset in which datasplits are allowed.  [0,0.5) values are allowed.
                           
        min_sample_split : int, optional
            Minimum number of points each cluster should contain selected by the user.
            
        **kernel_pca_args : 
            arguments 
            
        output_matrix : numpy.ndarray
            Model's step by step execution output.
            
        cluster_labels :
            Extracted clusters from the algorithm
        
    Attributes
    ----------
    
    """ 
    def __init__(self, reduction_method='pca', max_clusters_number = 100, bandwidth_scale = 0.5, percentile = 0.1, min_sample_split = 5, **kernel_pca_args):
        self.reduction_method = reduction_method
        self.max_clusters_number = max_clusters_number
        self.split_data_bandwidth_scale = bandwidth_scale
        self.percentile = percentile
        self.min_sample_split = min_sample_split
        self.kernel_pca_args = kernel_pca_args
    
    
    #%% Main algorithm execution methods
    def predict(self, X):
        """
        Create the PDDP tree and return the results of the dataset `X`, in the form of a dePDDP object.
        
        Parameters
        ----------
            X: numpy.ndarray
                data matrix (must check and return an error if not)
            
        Returns
        -------
        self : object
            A dePDDP class type object, with complete results on the algorithm's analysis
        """
        self.X = X
        self.samples_number = X.shape[0]
        
        # create an id vector for the samples of X
        indices = np.array([ int(i) for i in range(np.size(self.X, 0)) ])
        
        # initialize tree and root node                 # step (0)
        tree = Tree()
        # nodes unique IDs indicator
        self.node_ids = 0
        # nodes next color indicator (necessary for visualization purposes)
        self.cluster_color = 0
        tree.create_node(tag = 'cl_' + str(self.node_ids), identifier = self.node_ids, data = self.calculate_density(indices, self.X, self.cluster_color))
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
        
            self.split_function(tree, selected_node)               # step (1)
            
            # select the next kid for split based on the local minimum density
            selected_node = self.select_kid(tree.leaves())   # step (2)
            found_clusters = found_clusters +1                            # (ST1)
        
        
        self.tree = tree 
        return self
    
    
    def fit_predict(self, X):
        """
        Create the PDDP tree and return the results of the dataset `X`, in the form of a dePDDP object.
        
        Parameters
        ----------
        X : numpy.ndarray
            data matrix (numpy array, must check and return an error if not)
            
        Returns
        -------
        self : object
            A dePDDP class type object, with complete results on the algorithm's analysis
            
        """
        
        return self.predict(X)
    
    
    def split_function(self, tree, selected_node):
        """
        Split the indicated node on the minimum of the local minimum density of the data projected on the first principal component.
        
        Because python passes by refference data this function doesn't need a return statment.
        
        Parameters
        ----------
        tree : Tree object of treelib library
            The tree build by the dePDDP algorithm, in order to cluster the input data
        
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
        # Uses the calculate_density function to create the data for the node
        tree.create_node(tag = 'cl' + str(self.node_ids + 1), identifier = self.node_ids + 1, parent = node.identifier, data = self.calculate_density(left_kid_index, self.X[left_kid_index,:], node.data['color_key']))
        tree.create_node(tag = 'cl' + str(self.node_ids + 2), identifier = self.node_ids + 2, parent = node.identifier, data = self.calculate_density(right_kid_index, self.X[right_kid_index,:], self.cluster_color+1))
        
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
            A dePDDP class type object, with complete results on the algorithm's analysis
            
        """
        minimum_location = None
        
        # Remove the nodes that can not split further
        leaves = list(np.array(leaves)[ 
            [ True if not (i.data['split_density'] == None) else False for i in leaves ] 
        ])
        
        if len(leaves) > 0:
            for i in sorted(enumerate(leaves), key=lambda x:x[1].data['split_density']):
                if i[1].data['split_permition']:
                    minimum_location = i[1].identifier
                    break
        
        return minimum_location
    
    
    def execute_reduction_method(self, data_matrix):
        """
        Projection of the data matrix on to its principal components with utilization of the "Principal Components Analysis" and "Kernel Principal Components Analysis" methods.
        
        Parameters
        ----------
        data_matrix : ndarray of shape (n_samples, n_dims)
            The data matrix containing all the data for the samples
        
        Returns
        -------
        X_new : ndarray of shape (n_samples, 2)
            The projections of the samples on the first two components of the pca and kernel pca methods
            
        """
        if self.reduction_method == 'kpca':
            kernel_pca = KernelPCA(**self.kernel_pca_args)
            kernel_pca_X = kernel_pca.fit_transform(data_matrix)
            two_dimensions = kernel_pca_X[:,[0,1]]
        else:
            pca = PCA(n_components=2) # , svd_solver='full')
            two_dimensions = pca.fit_transform(data_matrix)
            
        return two_dimensions
    
    
    def calculate_density(self, indices, data_matrix, key):
        """
        Calculation of the projections on to the Principal Components with the utilization of the "Principal Components Analysis" and the "Kernel Principal Components Analysis" methods. 
        
        Determination of the projection's density and search for the local minima of the density. The lowest minima point within the allowed sample percetiles' of the projection's density.
        
        This function leads to the second Stopping criterion 2 of the algorithm. 
        
        Parameters
        ----------
        indices : ndarray of shape (n_samples,)
            The index of the samples in the original data matrix
            
        data_matrix : ndarray of shape (n_samples, n_dims)
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
            # execute pca on the data matrix
            two_dimensions = self.execute_reduction_method(data_matrix)
            one_dimension = two_dimensions[:,0]
            
            # calculate the standared deviation of the data 
            bandwidth = sm.nonparametric.bandwidths.select_bandwidth(one_dimension, "silverman", kernel=None)
            
            # calculate the density function on the 1st Princpal Component
            # x_ticks: projection points on the 1st PC
            # evaluation: the density of the projections on the 1st PC
            x_ticks, evaluation = FFTKDE(kernel='gaussian', bw = (self.split_data_bandwidth_scale * bandwidth)).fit(one_dimension).evaluate()
            # calculate all the local minima
            minimum_indices = argrelextrema(evaluation, np.less)[0]
            
            # find the location of the local minima and make sure they are with in the given percentile limits
            local_minimum_index = np.where(np.logical_and( x_ticks[minimum_indices] > np.quantile(one_dimension, self.percentile), x_ticks[minimum_indices] < np.quantile(one_dimension, (1-self.percentile)) ))
            
            # list all the numbers for the local minima (ee) and their respective position (ss) on the 1st PC
            ss = x_ticks[minimum_indices][local_minimum_index]
            ee = evaluation[minimum_indices][local_minimum_index]
            
            # if there is at least one local minima split the data
            if ss.size > 0:
                minimum_location = np.argmin(ee)
                splitpoint = ss[minimum_location]
                split_density = ee[minimum_location]
                flag = True
            else:
                splitpoint = None       # (ST2)
                split_density = None    # (ST2)
                flag = False            # (ST2)
        # =========================
        else:
            two_dimensions = None
            splitpoint = None           # (ST2)
            split_density = None        # (ST2)
            flag = False                # (ST2)
        
        return {'indices': indices, 'projection': two_dimensions, 'splitpoint': splitpoint, 'split_density': split_density, 'split_permition': flag, 'color_key': key}
    
    
   
    #%% Properties
    @property
    def reduction_method(self):
        return self._reduction_method
    
    @reduction_method.setter
    def reduction_method(self, v):
        if not (v in ['pca', 'kpca']):
            raise ValueError("reduction_method: " + str(v) + ": Unknown reduction method!")
        self._reduction_method = v
        
    @property
    def max_clusters_number(self):
        return self._max_clusters_number
    
    @max_clusters_number.setter
    def max_clusters_number(self, v):
        if v < 0 or (not isinstance(v, int)):
            raise ValueError("min_sample_split: Invalid value it should be int and > 1")
        self._max_clusters_number = v
    
    @property
    def split_data_bandwidth_scale(self):
        return self._split_data_bandwidth_scale
    
    @split_data_bandwidth_scale.setter
    def split_data_bandwidth_scale(self, v):
        if v > 1.0 and v <= 0:
            raise ValueError("split_data_bandwidth_scale: Should be between (0,1) interval")
        self._split_data_bandwidth_scale = v
    
    @property
    def percentile(self):
        return self._percentile
    
    @percentile.setter
    def percentile(self, v):
        if v >= 0.5 and v < 0:
            raise ValueError("percentile: Should be between [0,0.5) interval")
        self._percentile = v
    
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
    def cluster_labels(self):
        cluster_labels = np.ones(np.size(self.X, 0))
        for i in self.tree.leaves():
            cluster_labels[i.data['indices']] = i.identifier
        self.cluster_labels = cluster_labels
        return self._cluster_labels
    
    @cluster_labels.setter
    def cluster_labels(self, v):
        self._cluster_labels = v
    
    





