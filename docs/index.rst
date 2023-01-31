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
   :target: https://www.codacy.com/gh/panagiotisanagnostou/HiPart/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=panagiotisanagnostou/HiPart&amp;utm_campaign=Badge_Grade
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://github.com/panagiotisanagnostou/HiPart/blob/main/LICENSE

HiPart: Hierarchical divisive clustering toolbox
------------------------------------------------
HiPart is a package created for the implementation of hierarchical divisive clustering algorithms. Even among this family of algorithms, its specialty is high-performance algorithms for high-dimensional big data. It is a package with similar execution principles as the scikit-learn package. It also provides two types of static visualizations for all the algorithms executed in the package, with the addition of linkage generation for the divisive hierarchical clustering structure. Finally, the package provides an interactive visualization for manipulating the PDDP-based algorithms' split-point for each of the splits those algorithms generated from the clustering process.

Installation
------------
For the installation of the package, the only necessary actions and requirements are a version of Python higher or equal to 3.7 and the execution of the following command.

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

Users can find complete execution examples for all the algorithms of the HiPart package in the `clustering_example <https://github.com/panagiotisanagnostou/HiPart/blob/main/examples/clustering_example.py>`_ file of the repository. Also, the users can find a KernelPCA method usage example in the `clustering_with_kpca_example <https://github.com/panagiotisanagnostou/HiPart/blob/main/examples/clustering_with_kpca_example.py>`_ file of the repository. Finally, the file `interactive_visualization_example <https://github.com/panagiotisanagnostou/HiPart/blob/main/examples/interactive_visualization_example.py>`_ contains an example execution of the interactive visualization. The instructions for the interactive visualization GUI can be found with the execution of this visualization.


Citation
--------

	.. code-block:: bibtex
		
		@misc{anagnostou2022hipart,
		  doi = {10.48550/ARXIV.2209.08680},
		  url = {https://arxiv.org/abs/2209.08680},
		  author = {Anagnostou, Panagiotis and Tasoulis, Sotiris and Plagianakos, Vassilis and Tasoulis, Dimitris},
		  keywords = {Machine Learning (stat.ML), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
		  title = {HiPart: Hierarchical Divisive Clustering Toolbox},
		  publisher = {arXiv},
		  year = {2022},
		}


Acknowledgments
---------------
This project has received funding from the Hellenic Foundation for Research and Innovation (HFRI), under grant agreement No 1901.


Collaborators
-------------
Dimitris Tasoulis [:email:](d.tasoulis@thesignalgroup.com)
Panagiotis Anagnostou [:email:](panagno@uth.gr)
Sotiris Tasoulis [:email:](stasoulis@uth.gr)



.. toctree::
   :hidden:

   modules
   examples

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
