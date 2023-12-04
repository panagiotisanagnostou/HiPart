.. HiPart documentation master file, created by
   sphinx-quickstart on Sat Apr  9 18:10:05 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to HiPart's documentation!
==================================

.. image:: https://img.shields.io/pypi/v/HiPart.svg?color=blue
   :target: https://pypi.python.org/pypi/HiPart
.. image:: https://github.com/panagiotisanagnostou/HiPart/actions/workflows/python-app.yml/badge.svg
   :target: https://github.com/panagiotisanagnostou/HiPart/blob/main/.github/workflows/python-app.yml
.. image:: https://codecov.io/gh/panagiotisanagnostou/HiPart/branch/main/graph/badge.svg?token=FHoZrLjqfj
   :target: https://codecov.io/gh/panagiotisanagnostou/HiPart
.. image:: https://app.codacy.com/project/badge/Grade/60c751d914474e288b369461e6e3466a
   :target: https://app.codacy.com/gh/panagiotisanagnostou/HiPart/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://github.com/panagiotisanagnostou/HiPart/blob/main/LICENSE
.. image:: https://joss.theoj.org/papers/10.21105/joss.05024/status.svg
   :target: https://doi.org/10.21105/joss.05024

HiPart: Hierarchical divisive clustering toolbox
------------------------------------------------
HiPart is a package created for the implementation of hierarchical divisive clustering algorithms. Even among this family of algorithms, its specialty is high-performance algorithms for high-dimensional big data. It is a package with similar execution principles as the scikit-learn package. It also provides two types of static visualizations for all the algorithms executed in the package, with the addition of linkage generation for the divisive hierarchical clustering structure. Finally, the package provides an interactive visualization for manipulating the PDDP-based algorithms' split-point for each of the splits those algorithms generated from the clustering process.

Installation
------------
For the installation of the package, the only necessary actions and requirements are a version of Python higher or equal to 3.8 and the execution of the following command.

   .. code-block:: sh

      pip install HiPart

Simple Example Execution
------------------------
The example bellow is the simplest form of the package's execution. Shortly, it shows the creation of synthetic clustering dataset containing 6 clusters. Afterwards it is clustered with the dePDDP algorithm and only the cluster labels are returned.

   .. code-block:: py

      from HiPart.clustering import DePDDP
      from sklearn.datasets import make_blobs

      X, y = make_blobs(n_samples=1500, centers=6, random_state=0)

      clustered_class = DePDDP(max_clusters_number=6).fit_predict(X)

The HiPart package offers a comprehensive suite of examples to guide users in utilizing its various algorithms. These examples are conveniently located in the repository's examples directory.

For a general understanding of the package's capabilities, users can refer to the `clustering_example <https://github.com/panagiotisanagnostou/HiPart/blob/main/examples/clustering_example.py>`_ file. This file serves as a foundational guide, providing complete examples of the package's algorithms in action.

Additionally, for those interested in incorporating KernelPCA methods, the `clustering_with_kpca_example <https://github.com/panagiotisanagnostou/HiPart/blob/main/examples/clustering_with_kpca_example.py>`_ file is an invaluable resource. It offers a detailed example of how to apply KernelPCA within the context of the HiPart package.

Recognizing the importance of clustering via similarity or dissimilarity matrices, such as distance matrices, the HiPart package includes the `clustering_with_distance_matrix_example <https://github.com/panagiotisanagnostou/HiPart/blob/main/examples/distance_matrix_example.py>`_ file. This specific example demonstrates the use of the DePDDP algorithm with a distance matrix, offering a practical application scenario.

Lastly, the package features an interactive visualization component, which is exemplified in the `interactive_visualization_example <https://github.com/panagiotisanagnostou/HiPart/blob/main/examples/interactive_visualization_example.py>`_ file. This example not only showcases the execution of the interactive visualization but also provides comprehensive instructions for navigating the visualization GUI.

These resources collectively ensure that users of the HiPart package have a well-rounded and practical understanding of its functionalities and applications.


Citation
--------

   .. code-block:: bibtex

      @article{Anagnostou2023HiPart,
        title = {HiPart: Hierarchical Divisive Clustering Toolbox},
        author = {Panagiotis Anagnostou and Sotiris Tasoulis and Vassilis P. Plagianakos and Dimitris Tasoulis},
        year = {2023},
        journal = {Journal of Open Source Software},
        publisher = {The Open Journal},
        volume = {8},
        number = {84},
        pages = {5024},
        doi = {10.21105/joss.05024},
        url = {https://doi.org/10.21105/joss.05024}
      }


Acknowledgments
---------------
This project has received funding from the Hellenic Foundation for Research and Innovation (HFRI), under grant agreement No 1901.



Contents
-------------

.. toctree::
   :maxdepth: 3

   self
   HiPart
   examples

* :ref:`genindex`
* :ref:`modindex`
