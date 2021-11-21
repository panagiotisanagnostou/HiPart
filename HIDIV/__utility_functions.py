# -*- coding: utf-8 -*-
"""
Created on Tue Nov 7 15:05:21 2021

@author: Panagiotis Anagnostou
"""

import numpy as np
import warnings

from sklearn.decomposition import PCA, KernelPCA, FastICA

def execute_decomposition_method(data_matrix, decomposition_method, decomposition_args):
    """
    Projection of the data matrix onto its first two Components with the utilization of the "Principal Components Analysis", "Kernel Principal Components Analysis" or "Independent Component Analysis" decomposition methods.
    
    Parameters
    ----------
    data_matrix : numpy.ndarray
        The data matrix contains all the data for the samples.
    decomposition_method : str
        One of 'kpca', 'pca' and 'ica' the decomposition methods supported by this software.
    decomposition_args : dict
        Arguments to use by each of the decomposition methods utilized by the HIDIV package.
    
    Returns
    -------
    two_dimensions : numpy.ndarray
        The projections of the samples on the first two components of the pca and kernel pca methods.
        
    """
    
    if decomposition_method == 'kpca':
        kernel_pca = KernelPCA(n_components=2, **decomposition_args)
        two_dimensions = kernel_pca.fit_transform(data_matrix)
    elif decomposition_method == 'pca':
        pca = PCA(n_components=2, **decomposition_args)
        two_dimensions = pca.fit_transform(data_matrix)
    elif decomposition_method == 'ica':
        ica = FastICA(n_components=2, **decomposition_args)
        two_dimensions = ica.fit_transform(data_matrix)
    else:
        raise ValueError(": The dicomposition method (" + decomposition_method + ") is not supported!")
        
    return two_dimensions


def center_data(data):
    """
    Center the data on all its dimensions (subtract the mean of each variable, from each variable).
    
    Parameters
    ----------
    data : numpy.ndarray
        The data matrix containing all the data for the samples, samples are the rows and variables are the columns.
    
    Returns
    -------
    centered : numpy.ndarray
        The input data matrix centered on its variables.
        
    """
    
    # calculation of the mean of each variable (column)
    mean = np.nanmean(data, axis=0)
    # subtract the mean from each sample of the variable, for each variable separately
    centered = data - mean

    mean_1 = np.nanmean(centered, axis=0)
    # Verify that mean_1 is 'close to zero'. If X contains very large values, mean_1 can also be very large, due to a lack of precision of mean_. In this case, a pre-scaling of the concerned feature is efficient, for instance by its mean or maximum.
    if not np.allclose(mean_1, 0):
        warnings.warn(
            "Numerical issues were encountered when centering the data and might not be solved. Dataset may contain too large values. You may need to prescale your features."
        )
        centered -= mean_1
    
    return centered
