# -*- coding: utf-8 -*-
"""
Application of the Principal Direction Divisive Partitioning (PDDP).

@author: Panagiotis Anagnostou
"""

import graphviz as gv
import math
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm

from KDEpy import FFTKDE
from scipy.signal import argrelextrema
from sklearn.decomposition import PCA, KernelPCA
from tempfile import NamedTemporaryFile
from treelib import Tree

class dePDDP:
    """
    Class dePDDP. It executes the dePDDP algorithm 
    
    Parameters
    ----------
        max_clusters_number : int, optional
            desired number of cluster
            
        bandwidth_scale : float, optional
            standard deviation scaler for the density aproximation (0,1)
            
        percentile : float, optional
            the peprcentile of the entirety of the dataset in which datasplits are allowed.  [0,0.5) values are allowed.
                           
        min_sample_split : int, optional
            Minimum number of points each cluster should contain selected by the user
            
        **kernel_pca_args : 
            arguments 
            
        output_matrix : numpy.ndarray
            Model's execution output
            
        cluster_labels :
            Extracted clusters from the algorithm
        
    Attributes
    ----------
    
    """ 
    def __init__(self, reduction_method='pca', max_clusters_number = 100, bandwidth_scale = 0.5, percentile = 0.2, min_sample_split = 5, **kernel_pca_args):
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
        indices = np.array([ int(i) for i in range(np.size(self.X,0)) ])
        
        # initialize tree and root node                 # step (0)
        tree = Tree()
        # nodes unique IDs indicator
        self.node_ids = 0
        # nodes next color indicator (necessary for visualization purposes)
        self.cluster_color = 0
        tree.create_node(tag = 'cl_' + str(self.node_ids), identifier = self.node_ids, data = self.calden(indices, self.X, self.cluster_color))
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
        
            self.splitfun(tree, selected_node)               # step (1)
            
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
    
    
    def splitfun(self, tree, selected_node):
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
        left_kid_index = node.data['indices'][ np.where(node.data['projection'][:,0] >= node.data['splitpoint'])[0] ]
        # right child indecies
        right_kid_index = node.data['indices'][ np.where(node.data['projection'][:,0] < node.data['splitpoint'])[0] ]
        
        # Nodes and data creation for the children
        # Uses the calden function to create the data for the node
        tree.create_node(tag = 'cl' + str(self.node_ids + 1), identifier = self.node_ids + 1, parent = node.identifier, data = self.calden(left_kid_index, self.X[left_kid_index,:], node.data['color_key']))
        tree.create_node(tag = 'cl' + str(self.node_ids + 2), identifier = self.node_ids + 2, parent = node.identifier, data = self.calden(right_kid_index, self.X[right_kid_index,:], self.cluster_color+1))
        
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
            pca = PCA(n_components=2, svd_solver='full')
            two_dimensions = pca.fit_transform(data_matrix)
        return two_dimensions
    
    
    def calden(self, indices, data_matrix, key):
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
    
    
    #%% Manipulation of algorithms results methods
    def recalculate_after_spchange(self, split, splitpoint_value):
        """
        Given the serial number of the dePDDP tree`s internal nodes and a new splitpoint value recreate the results of the dePDDP algorithm with the indicated change.
        
        Parameters
        ----------
            split : int
                The serial number of the dePDDP tree`s internal nodes
            splitpoint_value : float
                New splitpoint value
            
        Returns
        -------
            self : object
                A dePDDP class type object, with complete results on the algorithm's analysis
        """
        
        tree = self.tree
        # find the cluster nodes
        clusters = tree.leaves()
        clusters = sorted(clusters, key = lambda x:x.identifier)
        
        # find the tree`s internal nodes a.k.a. splits
        dictionary_of_nodes = tree.nodes
        number_of_nodes = len(list(dictionary_of_nodes.keys()))
        leaf_node_list = [ j.identifier for j in clusters ]
        internal_nodes = [ i for i in range(number_of_nodes) if not (i in leaf_node_list) ]
        
        # insure that the root node will be always an internal node while is is the only node in the tree
        if len(internal_nodes) == 0: internal_nodes = [0]
        
        # remove all the splits (nodes) of the tree after the one bing manipulated.
        # this process starts from the last to the first to ensure deletion
        node_keys = list(dictionary_of_nodes.keys())
        for i in range(len(node_keys)-1, 0, -1):
            if node_keys[i] > internal_nodes[split]:
                if tree.get_node(node_keys[i]) != None:
                    if not tree.get_node(internal_nodes[split]).is_root():
                        if tree.parent(internal_nodes[split]).identifier != tree.parent(node_keys[i]).identifier:
                            tree.remove_node(node_keys[i])
                    else:
                        tree.remove_node(node_keys[i])
        
        # change the split permition of all the internal nodes to True so the algorithm can execute correctly
        dictionary_of_nodes = tree.nodes
        for i in dictionary_of_nodes:
            if dictionary_of_nodes[i].is_leaf():
                if dictionary_of_nodes[i].data['split_density'] != None:
                    dictionary_of_nodes[i].data['split_permition'] = True
        
        # reset status variables for the code to execute
        self.node_ids = len(list(dictionary_of_nodes.keys()))-1
        self.cluster_color = len(tree.leaves())+1
        tree.get_node(internal_nodes[split]).data['splitpoint'] = splitpoint_value
        
        # continuo the algorithm`s execution from the point left
        self.tree = tree
        self.tree = self.partial_predict()
        
        return self
    
    
    def delete_split(self, split):
        """
        Given the serial number of the dePDDP tree`s internal nodes permenanlty delete the subsplits that have it as parent in the dePDDP`s tree.
        
        Parameters
        ----------
            split : int
                The serial number of the dePDDP tree`s internal nodes
            
        Returns
        -------
            self : object
                A dePDDP class type object, with complete results on the algorithm's analysis
        """
        
        tree = self.tree
        # find the cluster nodes
        clusters = tree.leaves()
        clusters = sorted(clusters, key = lambda x:x.identifier)
        
        # find the tree`s internal nodes a.k.a. splits
        dictionary_of_nodes = tree.nodes
        number_of_nodes = len(list(dictionary_of_nodes.keys()))
        leaf_node_list = [ j.identifier for j in clusters ]
        internal_nodes = [ i for i in range(number_of_nodes) if not (i in leaf_node_list) ]
        
        # extract the node about to be deleted
        split_to_delete = internal_nodes[split]
        
        # delete the subtrees created by it`s children
        if len(tree.children(split_to_delete)) != 0:
            tree.remove_subtree(tree.children(split_to_delete)[0].identifier)
            tree.remove_subtree(tree.children(split_to_delete)[0].identifier)
        
        # change the split permition of node the split is indicating to False so the algorithm can execute without recreating it`s subpslits
        tree.get_node(split_to_delete).data['split_permition'] = False
        
        # re-serialize the identifiers numbering, the node tags and color_keys of tree`s nodes 
        tree = self.reserialize_node_elements(tree)
        
        return self
    
    
    #%% Visualization methods
    def split_viz_on_2_PCs(self, color_map="Set2"):
        # prepare the necessary data for this visualization
        dictionary_of_nodes, internal_nodes, _, sample_color = self.visulizezation_preparation(color_map)
        number_of_splits = len(internal_nodes)
        
        # insure that the root node will be always an internal node while is is the only node in the tree
        if number_of_splits == 0: internal_nodes = [0]
        
        # plot rows
        row_plots = 3 if math.ceil(number_of_splits) > 4 else 2
        
        # set figure size
        fig = plt.figure(figsize= (4 * row_plots, 3.2 * math.ceil(number_of_splits / row_plots)))
        
        # create grid for subfigures
        gs = gridspec.GridSpec(math.ceil(number_of_splits / row_plots) * 2, row_plots * 2, fig)
        
        for i in range(number_of_splits):
            # exrtact the subplot position
            row_from, row_to, col_from, col_to = self.grid_position(current = i, rows = row_plots, splits = number_of_splits, with_hist = False)
            
            # create the subplot on a pyplot axes
            ax = plt.subplot(gs[row_from:row_to, col_from:col_to])
            ax = self.make_simple_scatter(sp = ax, ndIdx = internal_nodes[i], splitPoint = dictionary_of_nodes[internal_nodes[i]].data['splitpoint'], PP = dictionary_of_nodes[internal_nodes[i]].data['projection'], sample_color = sample_color)
            
            # title the subplot
            ax.title.set_text("Original data with 1st split") if i == 0 else ax.title.set_text("Split no. " + str(i+1))
        
        fig.tight_layout()
        return fig
    
    
    
    def split_viz_with_density_margin(self, color_map = "Set2"):
        # prepare the necessary data for this visualization
        dictionary_of_nodes, internal_nodes, _, sample_color = self.visulizezation_preparation(color_map)
        number_of_splits = len(internal_nodes)
        
        # insure that the root node will be always an internal node while is is the only node in the tree
        if number_of_splits == 0: internal_nodes = [0]
        
        # plot subfigure
        row_plots = 3 if math.ceil(number_of_splits) > 4 else 2
        
        # set figure size
        fig = plt.figure(figsize=(4 * row_plots, 3.2 * math.ceil(number_of_splits / row_plots)), constrained_layout=True)
        
        # create grid for subfigures
        gs = gridspec.GridSpec(math.ceil(number_of_splits / row_plots) * 4, row_plots * 4, fig, hspace=.05)
        
        for i in range(number_of_splits):
            # exrtact the subplot position
            row_from, row_to, col_from, col_to = self.grid_position(current = i, rows = row_plots, splits = number_of_splits, with_hist = True)
            
            pr_col = sample_color[self.tree.get_node(internal_nodes[i]).data['indices']]
            principal_projections = dictionary_of_nodes[internal_nodes[i]].data['projection']
            splitPoint = dictionary_of_nodes[internal_nodes[i]].data['splitpoint']
            
            bandwidth = sm.nonparametric.bandwidths.select_bandwidth(principal_projections[:,0], "silverman", kernel=None)
            s, e = FFTKDE(kernel='gaussian', bw = (self.split_data_bandwidth_scale * bandwidth)).fit(principal_projections[:,0]).evaluate()
            
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
    
    
    def tree_text_viz(self, title=None, name=None, file=None, directory=None, format = "png", height = None, width = None):
        
        pt = gv.Digraph(name, filename = name, format = format,  node_attr = {'shape': 'record', "fontsize": "16pt"}, edge_attr={"arrowhead": "empty", "arrowsize": "0.75"})
        
        height = 2.9 * self.tree.depth() if height == None else height
        width = 4.5 * ((self.tree.depth() - 1) / 2) if width == None else width
        pt.attr(label = r"" + title + "\n", labelloc = "t", fontsize = "26pt", size = str(height) + "," + str(width) + "!", margin = "0")
        
        te = []
        ndIDs =  list(self.tree.nodes.keys())
        for i in ndIDs:
            # node data creation
            split_density = str(np.round(self.tree.get_node(i).data['split_density'], decimals=5)) if self.tree.get_node(i).data['split_density'] != None else "None"
            splitpoint = str(np.round(self.tree.get_node(i).data['splitpoint'], decimals=5)) if self.tree.get_node(i).data['splitpoint'] != None else "None"
            lIdx = str(len(self.tree.get_node(i).data['indices']))
            
            # nnode creation
            pt.node("nd_" + str(i), label=r"""<
                <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                <TR><TD>Node-ID: """+ str(i) +"""</TD></TR>
                <TR><TD>Min-density: """+ split_density +"""</TD></TR>
                <TR><TD>Slpit-point: """+ splitpoint +"""</TD></TR>
                <TR><TD>No of Objs: """+ lIdx +"""</TD></TR>
                </TABLE>
                >""")
            
            # create the node edges
            if i != 0:
                te.append(("nd_" + str(self.tree.parent(i).identifier), "nd_" + str(i)))
        # import the edges on the digraph
        pt.edges(te)
        pt.render()
        
        if file == None:
            file = name
        
        pt.save(filename=file, directory=directory)
        
        return pt
	
    
    def tree_data_viz(self, title=None, name=None, file=None, directory=None, format="png", color_map="tab20", width=None, height=None, rootLabelActive=True, rootLabelText="Root", nodeLabels=False, splitPointVisible=True):
        # prepare the necessary data for this visualization
        dictionary_of_nodes, internal_nodes, color_list, _ = self.visulizezation_preparation(color_map)
        
        # insure that the root node will be always an internal node while is is the only node in the tree
        if len(internal_nodes) == 0: internal_nodes = [0]
        
        if file == None:
            file = name
            
        pt = gv.Digraph(name, filename=file, directory=directory, format=format, node_attr={"shape": "plaintext", "labelloc": "t", "fontsize": "16pt", "margin": "0.8,1.18"}, edge_attr={"arrowhead": "empty", "arrowsize": "0.75"})
        
        width = 12 * ((self.tree.depth() - 1) / 2) if width == None else width
        height = 6.5 * (self.tree.depth() - 1) if height == None else height
        if title != None:
            pt.attr(label = r"" + title + "\n", labelloc = "t", fontsize = "26pt")
        pt.attr(size = str(height) + "," + str(width) + "!", margin = "0")
        
        # temp figures 
        temp_files = []
        te = []
        split = 0
        for i in internal_nodes:
            temp_files.append(NamedTemporaryFile("wb+", suffix='.svg', delete=False))
            
            # create the node`s name if needed
            if nodeLabels:
                ndName = 'Split ' + str(split)
                split += 1
            else:
                ndName = rootLabelText if i == 0 and rootLabelActive else ""
            
            node = self.tree.get_node(i)
            
            # mutch the smaples of each node with thier respective clusters
            if self.tree.depth() == 0:
                cluster_map = np.full(len(node.data['indices']), node.data['color_key'])
                
            else:
                cluster_map = np.full(len(node.data['indices']), self.tree.children(i)[0].data['color_key'])
                next_child_color = [ True if j in self.tree.children(i)[1].data['indices'] else False for j in node.data['indices'] ]
                cluster_map[next_child_color] = self.tree.children(i)[1].data['color_key']
            
            # color the samples based on cluster they belong to
            colors = np.zeros(len(node.data['indices']))
            colors = np.array([ color_list[int(j)] for j in cluster_map ])
            
            # create and save as a temp file the scatter plot that the node will use as input
            plt = self.simple_scatter_of_node(x=node.data['projection'][:,0], y=node.data['projection'][:,1], colors=colors, splitPoint=self.tree.get_node(i).data['splitpoint'], splitPointVisible=splitPointVisible)
            plt.savefig(temp_files[-1], bbox_inches='tight', pad_inches=0.18)
            plt.close()
            
            # create the node edges
            pt.node("nd_" + str(i), image=temp_files[-1].name, label=ndName)
            
            if i != 0:
                te.append(("nd_" + str(self.tree.parent(i).identifier), "nd_" + str(i)))
        
        # import the edges on the digraph
        pt.edges(te)
        
        pt.render()
        pt.save(filename=file, directory=directory)
        
        for i in temp_files:
            i.close()
            
        return pt


    #%% Support methods
    def partial_predict(self):
        tree = self.tree
        
        found_clusters = len(tree.leaves())
        selected_node = self.select_kid(tree.leaves())
        
        while selected_node != None and found_clusters < self.max_clusters_number:  # (ST1) or (ST2)
        
            self.splitfun(tree, selected_node)               # step (1)
            
            # select the next kid for split based on the local minimum density
            selected_node = self.select_kid(tree.leaves())   # step (2)
            found_clusters = found_clusters +1                            # (ST1)
            
        return tree
    
    
    def reserialize_node_elements(self, tree):
        dictionary_of_nodes = self.tree.nodes
        list_of_ids = list(dictionary_of_nodes.keys())
        
        # New tree creation for correct change of node names and values
        node_id = 0
        new_tree = Tree()
        new_tree.create_node(tag = 'cl' + str(node_id), identifier = node_id, data = dictionary_of_nodes[0].data)
        
        if len(dictionary_of_nodes.keys()) == 1:
            return tree
        
        correspondence_dictionary = {0:0}
        for i in range(1, len(list_of_ids)):
            node_id += 1
            correspondence_dictionary[self.tree.get_node(list_of_ids[i]).identifier] = node_id
            new_tree.create_node(tag = 'cl' + str(node_id), identifier = node_id, parent = correspondence_dictionary[tree.parent(list_of_ids[i]).identifier], data = dictionary_of_nodes[list_of_ids[i]].data)
        
        return new_tree
    
    
    def visulizezation_preparation(self, color_map):
        dictionary_of_nodes = self.tree.nodes
        
        # get colormap
        color_map = matplotlib.cm.get_cmap(color_map, len(list(dictionary_of_nodes.keys())))
        color_list = [ color_map(i) for i in range(color_map.N) ]
        
        # find the clusters in the data
        clusters = self.tree.leaves()
        clusters = sorted(clusters, key = lambda x:x.identifier)
        
        # create colormap for the generated clusters
        cluster_map = np.zeros(self.samples_number)
        for i in clusters:
            cluster_map[i.data['indices']] = i.identifier
        # color assignment
        sample_color = np.array([ color_list[int(i)] for i in cluster_map ])
        
        # find all the spliting points of the dataset via the internal nodes of
        # the tree
        number_of_nodes = len(list(dictionary_of_nodes.keys()))
        leaf_node_list = [ j.identifier for j in clusters ]
        internal_nodes = [ i for i in range(number_of_nodes) if not (i in leaf_node_list) ]
        
        return dictionary_of_nodes, internal_nodes, color_list, sample_color
    
    
    def grid_position(self, current, rows, splits, with_hist=True):
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
    
    
    def make_simple_scatter(self, sp, ndIdx, splitPoint, PP, sample_color):
        # color via the above colormap
        pr_col = sample_color[self.tree.get_node(ndIdx).data['indices']]
        
        # Create and add the subplot
        sp.scatter(PP[:,0], PP[:,1], c=pr_col, s=18, marker=".")
        sp.set_xticks([])
        sp.set_yticks([])
        sp.grid(False)
        sp.axvline(x=splitPoint, color='red', lw=1)
        sp.margins(0.03)
        
        return sp
    
    
    def simple_scatter_of_node(self, x, y, colors, splitPoint, splitPointVisible=True, name=None):
        
        plt.figure(figsize=(4, 2.75))
        plt.scatter(x, y, c=colors, s=18, marker='.')
        if splitPointVisible: 
            plt.axvline(x=splitPoint, color='red', lw=1)
        plt.margins(0.05)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.title(name)
        plt.subplots_adjust(left=-0.5, right=0.5, top=0.58, bottom=-0.42)
        
        return plt
    
    
    #%% Properties
    @property
    def reduction_method(self):
        return self._reduction_method
    
    @reduction_method.setter
    def reduction_method(self, v):
        if not (v in ['pca', 'kpca']):
            raise ValueError("reduction_method: " + str(v) + 
                             ": Unknown reduction method!")
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
    
    





