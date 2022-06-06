"""
    HiPart: Hierarchical divisive clustering toolbox

    HiPart is a package created for the implementation of hierarchical divisive
    clustering algorithms. Even among this family of algorithms, its specialty
    is high-performance algorithms for high-dimensional big data. It is a
    package with similar execution principles as the scikit-learn package. It
    also provides two types of static visualizations for all the algorithms
    executed in the package, with the addition of linkage generation for the
    divisive hierarchical clustering structure. Finally, the package provides
    an interactive visualization for manipulating the PDDP-based algorithms'
    split-point for each of the splits those algorithms generated from the
    clustering process.

"""

from KDEpy.NaiveKDE import NaiveKDE
from KDEpy.TreeKDE import TreeKDE
from KDEpy.FFTKDE import FFTKDE

__version__ = "0.1.16"
__author__ = "Panagiotis Anagnostou"

TreeKDE = TreeKDE
NaiveKDE = NaiveKDE
FFTKDE = FFTKDE
