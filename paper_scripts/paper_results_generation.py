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
Results generator.

@author: Panagiotis Anagnostou
"""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import __utilities as util
import csv
import gc
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")


def evaluate(X, y, name):
    print(X.shape)

    results = util.executor(X=X, y=y)

    with open("result_dict.dump", "rb") as ind:
        all_results = pickle.load(ind)

    all_results[name + str(X.shape)] = results
    del X, y

    with open("result_dict.dump", "wb") as out:
        pickle.dump(all_results, out)

    gc.collect()


if __name__ == "__main__":
    with open("result_dict.dump", "wb") as outf:
        pickle.dump(dict(), outf)

    # %% Analyze Baron dataset
    name = "DRComparison-Baron"
    X, y = util.h5file("data/", name)

    evaluate(X, y, name)

    # %% Analyze Deng dataset
    name = "mat-Deng"
    X, y = util.h5file("data/", name)

    evaluate(X, y, name)

    # %% Analyze Chen dataset
    name = "scRNAseq-ChenBrainData"
    X, y = util.h5file("data/", name)

    evaluate(X, y, name)

    # %% Analyze Cancer dataset
    name = "Cancer"
    X, y = util.h5file("data/", name)

    evaluate(X, y, name)

    # %% Analyze USPS dataset
    name = "USPS"
    X, y = util.h5file("data/", name)

    evaluate(X, y, name)

    # %% Analyze BBC dataset
    name = "BBC"
    X, y = util.h5file("data/", name)

    evaluate(X, y, name)

    # %% Generate final results
    with open("result_dict.dump", "rb") as ind:
        res = pickle.load(ind)

    algos = [
        "dePDDP",
        "bisect k-Means",
        "kM-PDDP",
        "PDDP",
        "iPDDP",
        "k-Means",
        "Fuzzy c-means",
        "Agglomerative",
        "OPTICS",
    ]

    with open("paper_results.csv", "w", encoding="UTF8", newline="") as f:
        for i in res:
            if not res[i] is None:
                if res[i][1][0] != 0:
                    writer = csv.writer(f, delimiter=";")

                    # write the header
                    writer.writerow([i] + list(np.full(3, "")))
                    writer.writerow(["Algorithm", "Execution Time", "MNI", "ARI"])

                    if res[i].shape[1] > 3:
                        merged = np.array(
                            [
                                [
                                    str(np.round(j[2 * k], decimals=2))
                                    + " ("
                                    + str(np.round(j[2 * k + 1], decimals=2))
                                    + ")"
                                    for k in range(3)
                                ]
                                for j in res[i]
                            ]
                        )
                    else:
                        merged = res[i]
                    data = np.concatenate(([[j] for j in algos], merged), axis=1)
                    # write multiple rows
                    writer.writerows(data)
