.. HiPart documentation master file, created by
   sphinx-quickstart on Sat Apr  9 18:10:05 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to HiPart's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules


HiPart: Hierarchical divisive clustering toolbox
------------------------------------------------
HiPart is a package created for the implementation of hierarchical divisive
Clustering algorithms. Even among this family of algorithms, its specialty is
algorithms with the highest time performance. It is a package with similar
execution principles as the scikit-learn package. It also provides two types of
static visualizations for all the algorithms executed in the package and
interactive visualization for the manipulation of the PDDP based algorithms'
split-point for each of the splits those algorithms generated from the input
data.




Installation
------------
For the installation of the package the only necessary actions and requirements
are a version of Python higher or equal to 3.7 and the execution of the
following command.

   .. code-block:: sh
   
      pip install HiPart



Simple Example Execution
------------------------
The example bellow is the simplest form of the package's execution. Shortly, it
shows the creation of synthetic clustering dataset containing 6 clusters.
Afterwards it is clustered with the dePDDP algorithm while each step of the
clustering is then visualized with some additional information based on the
dePDDP algorithm.

   .. code-block:: py

      from HiPart import visualizations as viz
      from HiPart.clustering import dePDDP
      from sklearn.datasets import make_blobs

      X, y = make_blobs(n_samples=1500, centers=6, random_state=41179)
      print("Example data shape: {}\n".format(X.shape))

      clustered_class = dePDDP(max_clusters_number=6).fit(X)

      viz.split_visualization(clustered_class).show()



Collaborators
-------------
Dimitris Tasoulis (d.tasoulis@thesignalgroup.com)

Sotiris Tasoulis (stasoulis@uth.gr)


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
