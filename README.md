[![PyPI](https://img.shields.io/pypi/v/HiPart?color=blue)](https://pypi.org/project/HiPart/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/HiPart)](https://pypi.org/project/HiPart/)
[![example workflow](https://github.com/panagiotisanagnostou/HiPart/actions/workflows/python-app.yml/badge.svg)](https://github.com/panagiotisanagnostou/HiPart/blob/main/.github/workflows/python-app.yml)
[![codecov](https://codecov.io/gh/panagiotisanagnostou/HiPart/branch/main/graph/badge.svg?token=FHoZrLjqfj)](https://codecov.io/gh/panagiotisanagnostou/HiPart)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/60c751d914474e288b369461e6e3466a)](https://www.codacy.com/gh/panagiotisanagnostou/HiPart/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=panagiotisanagnostou/HiPart&amp;utm_campaign=Badge_Grade)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/panagiotisanagnostou/HiPart/blob/main/LICENSE)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05024/status.svg)](https://doi.org/10.21105/joss.05024)

HiPart: Hierarchical divisive clustering toolbox
================================================
This repository presents the HiPart package, an open-source native python library that provides efficient and interpretable implementations of divisive hierarchical clustering algorithms. HiPart supports interactive visualizations for the manipulation of the execution steps allowing the direct intervention of the clustering outcome. This package is highly suited for Big Data applications as the focus has been given to the computational efficiency of the implemented clustering methodologies. The dependencies used are either Python build-in packages or highly maintained stable external packages. The software is provided under the MIT license.

Installation
------------
For the installation of the package, the only necessary actions and requirements are a version of Python higher or equal to 3.8 and the execution of the following command.

```bash
pip install HiPart
```

Simple Example Execution
------------------------
The example bellow is the simplest form of the package's execution. Shortly, it shows the creation of synthetic clustering dataset containing 6 clusters. Afterwards it is clustered with the DePDDP algorithm and only the cluster labels are returned.

```python
from HiPart.clustering import DePDDP
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=1500, centers=6, random_state=0)

clustered_class = DePDDP(max_clusters_number=6).fit_predict(X)
```

Users can find complete execution examples for all the algorithms of the HiPart package in the [clustering_example](https://github.com/panagiotisanagnostou/HiPart/blob/main/examples/clustering_example.py) file of the repository. Also, the users can find a KernelPCA method usage example in the [clustering_with_kpca_example](https://github.com/panagiotisanagnostou/HiPart/blob/main/examples/clustering_with_kpca_example.py) file of the repository. Finally, the file [interactive_visualization_example](https://github.com/panagiotisanagnostou/HiPart/blob/main/examples/interactive_visualization_example.py) contains an example execution of the interactive visualization. The instructions for the interactive visualization GUI can be found with the execution of this visualization.

Documentation
-------------
The full documentation of the package can be found [here](https://hipart.readthedocs.io).

Citation
--------

```bibtex
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
```

Acknowledgments
---------------
This project has received funding from the Hellenic Foundation for Research and Innovation (HFRI), under grant agreement No 1901.

Collaborators
-------------
Dimitris Tasoulis [:email:](mailto:d.tasoulis@thesignalgroup.com)
Panagiotis Anagnostou [:email:](mailto:panagno@uth.gr)
Sotiris Tasoulis [:email:](mailto:stasoulis@uth.gr)
Vassilis Plagianakos [:email:](mailto:vpp@uth.gr)
