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
Utility functions for the paper_results_generation.

@author: Panagiotis Anagnostou
"""

try:
    from fcmeans import FCM
except ImportError:
    raise ImportError('Install the package "fuzzy-c-means"!')
from sklearn.cluster import KMeans, OPTICS, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
from HiPart.clustering import bicecting_kmeans
from HiPart.clustering import dePDDP
from HiPart.clustering import iPDDP
from HiPart.clustering import kM_PDDP
from HiPart.clustering import PDDP

try:
    import h5py
except ImportError:
    raise ImportError('Install the package "h5py"!')
import numpy as np
import re
import string
import time
import traceback
import warnings

warnings.filterwarnings("ignore")


def read_data_file(name):
    data = np.array([["0", "0", "0"]])
    with open(name, "r", encoding="utf8") as inf:
        for i in inf:
            i = re.sub(r"\n", "", i)
            temp = np.array(i.split("|")).transpose()
            if len(temp) > 3:
                temp = np.array([[temp[0], temp[1], "".join(temp[2:])]])
                data = np.concatenate((data, temp), axis=0)
            else:
                temp = np.array([temp])
                data = np.concatenate((data, temp), axis=0)
    return data[1:, :]


def doc_processecor(doc, stopwords):
    # remove urls
    tmp = re.sub(r"https?:\/\/", "", doc)
    # trurn to lower case
    tmp = tmp.lower()
    # remove punctuation
    tmp = "".join([i if i not in string.punctuation else " " for i in tmp])
    # applying function to the column
    tmp = tmp.split(" ")
    # remove stopwords
    tmp = [i for i in tmp if i not in stopwords]
    # make data a string
    return "".join(tmp)


# scale pixels
def prep_pixels(img_data):
    # convert from integers to floats
    img_norm = img_data.astype("float32")
    # normalize to range 0-1
    img_norm = img_norm / 255.0
    # return normalized images
    return img_norm


def execute_evaluation(X, y):
    cluster_number = len(np.unique(y))
    print("cluster_number= {}\n".format(cluster_number))

    results = np.zeros((9, 6))
    ffolds = 10
    sfolds = 3

    # dePDDP algorithm
    try:
        depddp_time = []
        depddp_mni = []
        depddp_ari = []
        for i in range(ffolds):
            depddp = dePDDP(
                max_clusters_number=cluster_number,
                bandwidth_scale=0.5,
                percentile=0.1,
            )
            tic = time.perf_counter()
            depddp = depddp.fit(X)
            toc = time.perf_counter()
            dePDDP_y = depddp.labels_

            depddp_time.append(toc - tic)
            depddp_mni.append(nmi(y, dePDDP_y))
            depddp_ari.append(ari(y, dePDDP_y))
        results[0, 0] = np.mean(depddp_time)
        results[0, 1] = np.std(depddp_time)

        results[0, 2] = np.mean(depddp_mni)
        results[0, 3] = np.std(depddp_mni)

        results[0, 4] = np.mean(depddp_ari)
        results[0, 5] = np.std(depddp_ari)

        print("depddp_time= {val:.5f}".format(val=np.mean(depddp_time)))
        print("depddp_mni= {val:.5f}".format(val=np.mean(depddp_mni)))
        print("depddp_ari= {val:.5f}\n".format(val=np.mean(depddp_ari)))
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        print("dePDDP: execution error!")

    # Bisecting kMeans algorithm execution
    try:
        bikmeans_time_l = []
        bikmeans_mni_l = []
        bikmeans_ari_l = []
        for i in range(ffolds):
            tic = time.perf_counter()
            bikmeans = bicecting_kmeans(max_clusters_number=cluster_number).fit(X)
            toc = time.perf_counter()
            bikmeans_time_l.append(toc - tic)
            bikmeans_mni_l.append(nmi(y, bikmeans.labels_))
            bikmeans_ari_l.append(ari(y, bikmeans.labels_))

        results[1, 0] = np.mean(bikmeans_time_l)
        results[1, 1] = np.std(bikmeans_time_l)

        results[1, 2] = np.mean(bikmeans_mni_l)
        results[1, 3] = np.std(bikmeans_mni_l)

        results[1, 4] = np.mean(bikmeans_ari_l)
        results[1, 5] = np.std(bikmeans_ari_l)

        print("bikmeans_time= {val:.5f}".format(val=np.mean(bikmeans_time_l)))
        print("bikmeans_mni= {val:.5f}".format(val=np.mean(bikmeans_mni_l)))
        print("bikmeans_ari= {val:.5f}\n".format(val=np.mean(bikmeans_ari_l)))
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        print("bicecting_kmeans: execution error!")

    # kM-PDDP algorithm execution
    try:
        kmpddp_time_l = []
        kmpddp_mni_l = []
        kmpddp_ari_l = []
        for i in range(ffolds):
            tic = time.perf_counter()
            kmpddp = kM_PDDP(max_clusters_number=cluster_number).fit(X)
            toc = time.perf_counter()
            kmpddp_time_l.append(toc - tic)
            kmpddp_mni_l.append(nmi(y, kmpddp.labels_))
            kmpddp_ari_l.append(ari(y, kmpddp.labels_))

        results[2, 0] = np.mean(kmpddp_time_l)
        results[2, 1] = np.std(kmpddp_time_l)

        results[2, 2] = np.mean(kmpddp_mni_l)
        results[2, 3] = np.std(kmpddp_mni_l)

        results[2, 4] = np.mean(kmpddp_ari_l)
        results[2, 5] = np.std(kmpddp_ari_l)

        print("kmpddp_time= {val:.5f}".format(val=np.mean(kmpddp_time_l)))
        print("kmpddp_mni= {val:.5f}".format(val=np.mean(kmpddp_mni_l)))
        print("kmpddp_ari= {val:.5f}\n".format(val=np.mean(kmpddp_ari_l)))
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        print("kM_PDDP: execution error!")

    # PDDP algorithm execution
    try:
        pddp_time_l = []
        pddp_mni_l = []
        pddp_ari_l = []
        for i in range(ffolds):
            tic = time.perf_counter()
            pddp = PDDP(max_clusters_number=cluster_number).fit(X)
            toc = time.perf_counter()
            pddp_time_l.append(toc - tic)
            pddp_mni_l.append(nmi(y, pddp.labels_))
            pddp_ari_l.append(ari(y, pddp.labels_))

        results[3, 0] = np.mean(pddp_time_l)
        results[3, 1] = np.std(pddp_time_l)

        results[3, 2] = np.mean(pddp_mni_l)
        results[3, 3] = np.std(pddp_mni_l)

        results[3, 4] = np.mean(pddp_ari_l)
        results[3, 5] = np.std(pddp_ari_l)

        print("pddp_time= {val:.5f}".format(val=np.mean(pddp_time_l)))
        print("pddp_mni= {val:.5f}".format(val=np.mean(pddp_mni_l)))
        print("pddp_ari= {val:.5f}\n".format(val=np.mean(pddp_ari_l)))
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        print("PDDP: execution error!")

    # iPDDP algorithm execution
    try:
        ipddp_time_l = []
        ipddp_mni_l = []
        ipddp_ari_l = []
        for i in range(ffolds):
            tic = time.perf_counter()
            ipddp = iPDDP(max_clusters_number=cluster_number).fit(X)
            toc = time.perf_counter()
            ipddp_time_l.append(toc - tic)
            ipddp_mni_l.append(nmi(y, ipddp.labels_))
            ipddp_ari_l.append(ari(y, ipddp.labels_))

        results[4, 0] = np.mean(ipddp_time_l)
        results[4, 1] = np.std(ipddp_time_l)

        results[4, 2] = np.mean(ipddp_mni_l)
        results[4, 3] = np.std(ipddp_mni_l)

        results[4, 4] = np.mean(ipddp_ari_l)
        results[4, 5] = np.std(ipddp_ari_l)

        print("ipddp_time= {val:.5f}".format(val=np.mean(ipddp_time_l)))
        print("ipddp_mni= {val:.5f}".format(val=np.mean(ipddp_mni_l)))
        print("ipddp_ari= {val:.5f}\n".format(val=np.mean(ipddp_ari_l)))
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        print("iPDDP: execution error!")

    # k-means algorithm
    try:
        kmeans_time_l = []
        kmeans_mni_l = []
        kmeans_ari_l = []
        for i in range(ffolds):
            kmeans = KMeans(cluster_number, algorithm="full")
            tic = time.perf_counter()
            kmeans_y = kmeans.fit_predict(X)
            toc = time.perf_counter()
            kmeans_time_l.append(toc - tic)
            kmeans_mni_l.append(nmi(y, kmeans_y))
            kmeans_ari_l.append(ari(y, kmeans_y))

        results[5, 0] = np.mean(kmeans_time_l)
        results[5, 1] = np.std(kmeans_time_l)

        results[5, 2] = np.mean(kmeans_mni_l)
        results[5, 3] = np.std(kmeans_mni_l)

        results[5, 4] = np.mean(kmeans_ari_l)
        results[5, 5] = np.std(kmeans_ari_l)

        print("kmeans_time= {val:.5f}".format(val=np.mean(kmeans_time_l)))
        print("kmeans_mni= {val:.5f}".format(val=np.mean(kmeans_mni_l)))
        print("kmeans_ari= {val:.5f}\n".format(val=np.mean(kmeans_ari_l)))
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        print("kmeans: execution error!")

    # fuzzy c means algorithm
    try:
        fcm_time_l = []
        fcm_mni_l = []
        fcm_ari_l = []
        for i in range(ffolds):
            fcm = FCM(n_clusters=cluster_number)
            tic = time.perf_counter()
            fcm.fit(X)
            fcm_y = fcm.predict(X)
            toc = time.perf_counter()
            fcm_time_l.append(toc - tic)
            fcm_mni_l.append(nmi(y, fcm_y))
            fcm_ari_l.append(ari(y, fcm_y))

        results[6, 0] = np.mean(fcm_time_l)
        results[6, 1] = np.std(fcm_time_l)

        results[6, 2] = np.mean(fcm_mni_l)
        results[6, 3] = np.std(fcm_mni_l)

        results[6, 4] = np.mean(fcm_ari_l)
        results[6, 5] = np.std(fcm_ari_l)

        print("fcm_time= {val:.5f}".format(val=np.mean(fcm_time_l)))
        print("fcm_mni= {val:.5f}".format(val=np.mean(fcm_mni_l)))
        print("fcm_ari= {val:.5f}\n".format(val=np.mean(fcm_ari_l)))
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        print("FCM: execution error!")

    # Agglomerative algorithm
    try:
        agg_time_l = []
        agg_mni_l = []
        agg_ari_l = []
        for i in range(sfolds):
            agg = AgglomerativeClustering(n_clusters=cluster_number)
            tic = time.perf_counter()
            agg_y = agg.fit_predict(X)
            toc = time.perf_counter()
            agg_time_l.append(toc - tic)
            agg_mni_l.append(nmi(y, agg_y))
            agg_ari_l.append(ari(y, agg_y))

        results[7, 0] = np.mean(agg_time_l)
        results[7, 1] = np.std(agg_time_l)

        results[7, 2] = np.mean(agg_mni_l)
        results[7, 3] = np.std(agg_mni_l)

        results[7, 4] = np.mean(agg_ari_l)
        results[7, 5] = np.std(agg_ari_l)

        print("agg_time= {val:.5f}".format(val=np.mean(agg_time_l)))
        print("agg_mni= {val:.5f}".format(val=np.mean(agg_mni_l)))
        print("agg_ari= {val:.5f}\n".format(val=np.mean(agg_ari_l)))
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        print("AgglomerativeClustering: execution error!")

    # OPTICS algorithm
    try:
        optics_time_l = []
        optics_mni_l = []
        optics_ari_l = []
        for i in range(sfolds):
            optics = OPTICS(n_jobs=-1)
            tic = time.perf_counter()
            optics_y = optics.fit_predict(X)
            toc = time.perf_counter()
            optics_time_l.append(toc - tic)
            optics_mni_l.append(nmi(y, optics_y))
            optics_ari_l.append(ari(y, optics_y))

        results[8, 0] = np.mean(optics_time_l)
        results[8, 1] = np.std(optics_time_l)

        results[8, 2] = np.mean(optics_mni_l)
        results[8, 3] = np.std(optics_mni_l)

        results[8, 4] = np.mean(optics_ari_l)
        results[8, 5] = np.std(optics_ari_l)

        print("optics_time= {val:.5f}".format(val=np.mean(optics_time_l)))
        print("optics_mni= {val:.5f}".format(val=np.mean(optics_mni_l)))
        print("optics_ari= {val:.5f}\n".format(val=np.mean(optics_ari_l)))
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        print("OPTICS: execution error!")

    return results


def h5file(data_folder, name):
    print("\n", name)

    f = h5py.File(data_folder + name + ".h5", "r")
    inData = f["data"]["matrix"][:].transpose()
    inTarget = f["class"]["categories"][:]
    inTarget = np.int32(inTarget) - 1

    if inData.shape[0] != len(inTarget):
        inData = inData.transpose()
        if inData.shape[0] != len(inTarget):
            print("Data ", name, "error! Pls Check!")
            f.close()
            return
    f.close()

    return inData, inTarget
