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

__version__ = "0.4.2"
__author__ = "Panagiotis Anagnostou"

TreeKDE = TreeKDE
NaiveKDE = NaiveKDE
FFTKDE = FFTKDE
