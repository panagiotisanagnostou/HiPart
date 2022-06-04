# -*- coding: utf-8 -*-
"""
Results generator.

@author: Panagiotis Anagnostou
"""

import __utilities as util
import gc
import numpy as np
import pandas as pd
import pickle
import re
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # %% Analyze Baron dataset
    name = "DRComparison-Baron"
    X, y = util.h5file("data/", name)

    print(X.shape)

    results = util.execute_evaluation(X=X, y=y)

    with open("result_dict.dump", "rb") as ind:
        all_results = pickle.load(ind)

    all_results[name + str(X.shape)] = results
    del X, y

    with open("result_dict.dump", "wb") as out:
        pickle.dump(all_results, out)

    gc.collect()

    # %% Analyze Baron dataset
    name = "mat-Deng"
    X, y = util.h5file("data/", name)

    print(X.shape)

    results = util.execute_evaluation(X=X, y=y)

    with open("result_dict.dump", "rb") as ind:
        all_results = pickle.load(ind)

    all_results[name + str(X.shape)] = results
    del X, y

    with open("result_dict.dump", "wb") as out:
        pickle.dump(all_results, out)

    gc.collect()

    # %% Analyze Baron dataset
    name = "scRNAseq-ChenBrainData"
    X, y = util.h5file("data/", name)

    print(X.shape)

    results = util.execute_evaluation(X=X, y=y)

    with open("result_dict.dump", "rb") as ind:
        all_results = pickle.load(ind)

    all_results[name + str(X.shape)] = results
    del X, y

    with open("result_dict.dump", "wb") as out:
        pickle.dump(all_results, out)

    gc.collect()

    # %% Analyze cancer dataset
    X = pd.read_csv("./data/cancer/data.csv", index_col=0, header=0)
    X = np.asarray(X, dtype="float64")
    y = pd.read_csv("./data/cancer/labels.csv", index_col=0, header=0)
    y["Class"] = pd.Categorical(y["Class"])
    y["Class"] = y.Class.cat.codes
    y = np.asarray(y).transpose()[0]

    print("\ncancer")
    print(X.shape)

    results = util.execute_evaluation(X=X, y=y)

    with open("result_dict.dump", "rb") as ind:
        all_results = pickle.load(ind)

    all_results["cancer " + str(X.shape)] = results
    del X, y

    with open("result_dict.dump", "wb") as out:
        pickle.dump(all_results, out)

    gc.collect()

    # %% Analyze usps dataset
    X = np.array(pd.read_csv("data/USPS-data", sep=";").applymap(lambda x: re.sub(",", ".", x))).astype(dtype="float32")
    y = np.array(pd.read_csv("data/USPS-class")).transpose()[0]

    print("\nusps")
    print(X.shape)

    results = util.execute_evaluation(X=X, y=y)

    with open("result_dict.dump", "rb") as ind:
        all_results = pickle.load(ind)

    all_results["usps " + str(X.shape)] = results
    del X, y

    with open("result_dict.dump", "wb") as out:
        pickle.dump(all_results, out)

    gc.collect()

    # %% Analyze bbc dataset
    X = np.array(pd.read_csv("data/bbc_data.csv", header=None))
    y = np.array(pd.read_csv("data/bbc_class.csv")).transpose()[0]

    print("\nbbc")
    print(X.shape)

    results = util.execute_evaluation(X=X, y=y)

    with open("result_dict.dump", "rb") as ind:
        all_results = pickle.load(ind)

    all_results["bbc " + str(X.shape)] = results
    del X, y

    with open("result_dict.dump", "wb") as out:
        pickle.dump(all_results, out)

    gc.collect()
