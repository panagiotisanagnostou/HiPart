from __future__ import unicode_literals
from distutils import dir_util
from HiPart import __utility_functions as uf
from HiPart import visualizations as viz
from HiPart.clustering import DePDDP
from HiPart.clustering import IPDDP
from HiPart.clustering import KMPDDP
from HiPart.clustering import PDDP
from HiPart.clustering import BisectingKmeans
from HiPart.clustering import MDH
from scipy.spatial import distance_matrix

import numpy as np
import os
import pickle
import pytest


@pytest.fixture
def datadir(tmpdir, request):
    """
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    """
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir


def test_depddp_return_type(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    new_obj = DePDDP(max_clusters_number=3).fit(data_import["data"])
    assert  isinstance(new_obj, DePDDP)


def test_ipddp_return_type(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    new_obj = IPDDP(max_clusters_number=3).fit(data_import["data"])
    assert isinstance(new_obj, IPDDP)


def test_kmpddp_return_type(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    new_obj = KMPDDP(max_clusters_number=3).fit(data_import["data"])
    assert isinstance(new_obj, KMPDDP)


def test_pddp_return_type(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    new_obj = PDDP(max_clusters_number=3).fit(data_import["data"])
    assert isinstance(new_obj, PDDP)


def test_bicecting_kmeans_return_type(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    new_obj = BisectingKmeans(max_clusters_number=3).fit(data_import["data"])
    assert isinstance(new_obj, BisectingKmeans)


def test_mdh_return_type(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    new_obj = MDH(
        max_clusters_number=3,
        random_state=0,
    ).fit(data_import["data"])
    assert isinstance(new_obj, MDH)


def test_depddp_parameter_errors():
    success_score = 0

    algorithm = DePDDP()
    success_score += 1 if isinstance(algorithm.bandwidth_scale, float) else 0
    success_score += 1 if isinstance(algorithm.percentile, float) else 0

    try:
        DePDDP(bandwidth_scale=-5)
    except Exception:
        success_score += 1
    try:
        DePDDP(percentile=.8)
    except Exception:
        success_score += 1

    assert success_score == 4


def test_ipddp_parameter_errors(datadir):
    success_score = 0

    algorithm = IPDDP()
    success_score += 1 if isinstance(algorithm.percentile, float) else 0

    try:
        IPDDP(percentile=.8)
    except Exception:
        success_score += 1

    assert success_score == 2


def test_kmpddp_parameter_errors(datadir):
    success = 0

    algorithm = KMPDDP(random_state=123)
    success += 1 if isinstance(algorithm.random_state, int) else 0

    try:
        KMPDDP(random_state=.8)
    except Exception:
        success += 1

    assert success == 2


def test_pddp_parameter_errors(datadir):
    success_score = 0

    algorithm = PDDP()
    success_score += 1 if isinstance(algorithm.decomposition_method, str) else 0
    success_score += 1 if isinstance(algorithm.max_clusters_number, int) else 0
    success_score += 1 if isinstance(algorithm.min_sample_split, int) else 0
    success_score += 1 if isinstance(algorithm.visualization_utility, bool) else 0
    success_score += 1 if isinstance(algorithm.distance_matrix, bool) else 0

    try:
        PDDP(decomposition_method="abc")
    except Exception:
        success_score += 1
    try:
        PDDP(max_clusters_number=-5)
    except Exception:
        success_score += 1
    try:
        PDDP(min_sample_split=-5)
    except Exception:
        success_score += 1
    try:
        PDDP(visualization_utility=5)
    except Exception:
        success_score += 1
    try:
        tmp = PDDP(
            decomposition_method="tsne",
        )
        tmp.visualization_utility = True
    except Exception:
        success_score += 1
    try:
        PDDP(distance_matrix=5)
    except Exception:
        success_score += 1
    try:
        obj = PDDP()
        obj.output_matrix = np.array([1, 2, 3])
    except Exception:
        success_score += 1
    try:
        obj = PDDP()
        obj.labels_ = np.array([1, 2, 3])
    except Exception:
        success_score += 1

    assert success_score == 13


def test_bicecting_kmeans_parameter_errors():
    success_score = 0

    algorithm = BisectingKmeans(random_state=5)
    success_score += 1 if isinstance(algorithm.random_state, int) else 0

    try:
        BisectingKmeans(random_state=.8)
    except Exception:
        success_score += 1

    assert success_score == 2


def test_mdh_parameter_errors():
    success_score = 0

    algorithm = MDH(random_state=5)
    success_score += 1 if isinstance(algorithm.max_iterations, int) else 0
    success_score += 1 if isinstance(algorithm.k, float) else 0
    success_score += 1 if isinstance(algorithm.percentile, float) else 0
    success_score += 1 if isinstance(algorithm.random_state, int) else 0

    try:
        MDH(max_iterations=-5)
    except Exception:
        success_score += 1
    try:
        MDH(k=-.8)
    except Exception:
        success_score += 1
    try:
        MDH(percentile=.8)
    except Exception:
        success_score += 1
    try:
        MDH(random_state=.8)
    except Exception:
        success_score += 1

    assert success_score == 8


def test_depddp_labels__return_type_and_form(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    results = DePDDP(max_clusters_number=3).fit_predict(data_import["data"])
    assert isinstance(results, np.ndarray) and results.ndim == 1


def test_ipddp_labels__return_type_and_form(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    results = IPDDP(max_clusters_number=3).fit_predict(data_import["data"])
    assert isinstance(results, np.ndarray) and results.ndim == 1


def test_kmpddp_labels__return_type_and_form(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    results = KMPDDP(max_clusters_number=3).fit_predict(data_import["data"])
    assert isinstance(results, np.ndarray) and results.ndim == 1


def test_pddp_labels__return_type_and_form(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    results = PDDP(max_clusters_number=3).fit_predict(data_import["data"])
    assert isinstance(results, np.ndarray) and results.ndim == 1


def test_bicecting_kmeans_labels_return_type_and_form(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    results = BisectingKmeans(
        max_clusters_number=3,
    ).fit_predict(data_import["data"])
    assert isinstance(results, np.ndarray) and results.ndim == 1


def test_mdh_labels_return_type_and_form(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    results = MDH(
        max_clusters_number=3,
        random_state=0,
    ).fit_predict(data_import["data"])
    assert isinstance(results, np.ndarray) and results.ndim == 1


def test_mdh_projections(datadir):
    np.random.seed(0)
    data = np.random.normal(size=(100, 2), loc=0, scale=0.001)

    results = MDH(
        max_clusters_number=3,
        random_state=0,
    ).fit_predict(data)
    assert isinstance(results, np.ndarray) and results.ndim == 1


def test_depddp_distance_matrix_executions(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    dist_matrix = distance_matrix(data_import["data"], data_import["data"])

    success_score = 0

    try:
        DePDDP(
            decomposition_method="mds",
            max_clusters_number=3,
            distance_matrix=False,
        ).fit(dist_matrix)
    except Exception:
        success_score += 1
    try:
        DePDDP(
            decomposition_method="pca",
            max_clusters_number=3,
            distance_matrix=True,
        ).fit(dist_matrix)
    except Exception:
        success_score -= 1
    try:
        tmp = DePDDP(
            decomposition_method="pca",
            max_clusters_number=3,
            distance_matrix=True,
        )
        tmp.decomposition_method = "pca"
        tmp.fit(dist_matrix)
    except Exception:
        success_score += 1
    try:
        DePDDP(
            decomposition_method="mds",
            max_clusters_number=3,
            distance_matrix=True,
        ).fit(data_import["data"])
    except Exception:
        success_score += 1

    assert success_score == 3


def test_ipddp_distance_matrix_executions(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    dist_matrix = distance_matrix(data_import["data"], data_import["data"])

    success_score = 0

    try:
        IPDDP(
            decomposition_method="mds",
            max_clusters_number=3,
            distance_matrix=False,
        ).fit(dist_matrix)
    except Exception:
        success_score += 1
    try:
        IPDDP(
            decomposition_method="pca",
            max_clusters_number=3,
            distance_matrix=True,
        ).fit(dist_matrix)
    except Exception:
        success_score -= 1
    try:
        tmp = IPDDP(
            decomposition_method="pca",
            max_clusters_number=3,
            distance_matrix=True,
        )
        tmp.decomposition_method = "pca"
        tmp.fit(dist_matrix)
    except Exception:
        success_score += 1
    try:
        IPDDP(
            decomposition_method="mds",
            max_clusters_number=3,
            distance_matrix=True,
        ).fit(data_import["data"])
    except Exception:
        success_score += 1

    assert success_score == 3


def test_kmpddp_distance_matrix_executions(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    dist_matrix = distance_matrix(data_import["data"], data_import["data"])

    success_score = 0

    try:
        KMPDDP(
            decomposition_method="mds",
            max_clusters_number=3,
            distance_matrix=False,
        ).fit(dist_matrix)
    except Exception:
        success_score += 1
    try:
        KMPDDP(
            decomposition_method="pca",
            max_clusters_number=3,
            distance_matrix=True,
        ).fit(dist_matrix)
    except Exception:
        success_score -= 1
    try:
        tmp = KMPDDP(
            decomposition_method="pca",
            max_clusters_number=3,
            distance_matrix=True,
        )
        tmp.decomposition_method = "pca"
        tmp.fit(dist_matrix)
    except Exception:
        success_score += 1
    try:
        KMPDDP(
            decomposition_method="mds",
            max_clusters_number=3,
            distance_matrix=True,
        ).fit(data_import["data"])
    except Exception:
        success_score += 1

    assert success_score == 3


def test_pddp_distance_matrix_executions(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    dist_matrix = distance_matrix(data_import["data"], data_import["data"])

    success_score = 0

    try:
        PDDP(
            decomposition_method="mds",
            max_clusters_number=3,
            distance_matrix=False,
        ).fit(dist_matrix)
    except Exception:
        success_score += 1
    try:
        PDDP(
            decomposition_method="pca",
            max_clusters_number=3,
            distance_matrix=True,
        ).fit(dist_matrix)
    except Exception:
        success_score -= 1
    try:
        tmp = PDDP(
            decomposition_method="pca",
            max_clusters_number=3,
            distance_matrix=True,
        )
        tmp.decomposition_method = "pca"
        tmp.fit(dist_matrix)
    except Exception:
        success_score += 1
    try:
        PDDP(
            decomposition_method="mds",
            max_clusters_number=3,
            distance_matrix=True,
        ).fit(data_import["data"])
    except Exception:
        success_score += 1

    assert success_score == 3


# scikit-learn's KMeans algorithm has a bad implermentation of the random_state
# parameter, so the results are not reproducible. This is why we cannot test
# the results of the BisectingKmeans algorithm.
# def test_bicecting_kmeans_results(datadir):
#     with open(datadir.join('test_data.dump'), "rb") as inf:
#         data_import = pickle.load(inf)
#
#     matrix_control = data_import["BisectingKmeans"]
#
#     matrix_test = BisectingKmeans(
#         max_clusters_number=3,
#         random_state=0,
#     ).fit(data_import["data"]).output_matrix
#     assert np.sum(matrix_test == matrix_control) == 1000


def test_mdh_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["MDH"]

    matrix_test = MDH(
        max_clusters_number=3,
        random_state=0,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 1000


def test_depddp_pca_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["DePDDP_pca"]

    matrix_test = DePDDP(
        max_clusters_number=3,
        random_state=0,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 1000


def test_ipddp_pca_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["IPDDP_pca"]

    matrix_test = IPDDP(
        max_clusters_number=3,
        random_state=0,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 1000


def test_kmpddp_pca_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["KMPDDP_pca"]

    matrix_test = KMPDDP(
        max_clusters_number=3,
        random_state=0,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 1000


def test_pddp_pca_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["PDDP_pca"]

    matrix_test = PDDP(
        max_clusters_number=3,
        random_state=0,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 1000


def test_depddp_ica_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["DePDDP_ica"]

    matrix_test = DePDDP(
        decomposition_method="ica",
        max_clusters_number=3,
        random_state=0,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 1000


def test_ipddp_ica_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["IPDDP_ica"]

    matrix_test = IPDDP(
        decomposition_method="ica",
        max_clusters_number=3,
        random_state=0,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 1000


# scikit-learn's KMeans algorithm has a bad implermentation of the random_state
# parameter, so the results are not reproducible. This is why we cannot test
# the results of the kmPDDP algorithm with ica.
# def test_kmpddp_ica_results(datadir):
#     with open(datadir.join('test_data.dump'), "rb") as inf:
#         data_import = pickle.load(inf)
#
#     matrix_control = data_import["KMPDDP_ica"]
#
#     matrix_test = KMPDDP(
#         decomposition_method="ica",
#         max_clusters_number=3,
#         random_state=0,
#     ).fit(data_import["data"]).output_matrix
#     assert np.sum(matrix_test == matrix_control) == 1000


def test_pddp_ica_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["PDDP_ica"]

    matrix_test = PDDP(
        decomposition_method="ica",
        max_clusters_number=3,
        random_state=0,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 1000


def test_depddp_kpca_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["DePDDP_kpca"]

    matrix_test = DePDDP(
        decomposition_method="kpca",
        max_clusters_number=3,
        random_state=0,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 1000


def test_ipddp_kpca_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["IPDDP_kpca"]

    matrix_test = IPDDP(
        decomposition_method="kpca",
        max_clusters_number=3,
        random_state=0,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 1000


def test_kmpddp_kpca_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["KMPDDP_kpca"]

    matrix_test = KMPDDP(
        decomposition_method="kpca",
        max_clusters_number=3,
        random_state=0,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 1000


def test_pddp_kpca_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["PDDP_kpca"]

    matrix_test = PDDP(
        decomposition_method="kpca",
        max_clusters_number=3,
        random_state=0,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 1000


def test_depddp_tsne_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["DePDDP_tsne"]

    matrix_test = DePDDP(
        decomposition_method="tsne",
        max_clusters_number=3,
        random_state=0,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 1000


def test_ipddp_tsne_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["IPDDP_tsne"]

    matrix_test = IPDDP(
        decomposition_method="tsne",
        max_clusters_number=3,
        random_state=0,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 1000


def test_kmpddp_tsne_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["KMPDDP_tsne"]

    matrix_test = KMPDDP(
        decomposition_method="tsne",
        max_clusters_number=3,
        random_state=0,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 1000


def test_pddp_tsne_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["PDDP_tsne"]

    matrix_test = PDDP(
        decomposition_method="tsne",
        max_clusters_number=3,
        random_state=0,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 1000


def test_pddp_mds_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    dist_matrix = distance_matrix(data_import["data"], data_import["data"])

    matrix_control = data_import["PDDP_mds"]

    matrix_test = PDDP(
        decomposition_method="mds",
        max_clusters_number=3,
        distance_matrix=True,
        random_state=0,
    ).fit(dist_matrix).output_matrix
    assert np.sum(matrix_test == matrix_control) == 1000


def test_split_visualization_plot_1(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    clustering = DePDDP(max_clusters_number=3).fit(data_import["data"])
    try:
        viz.split_visualization(clustering)
        assert True
    except Exception:
        assert False


def test_split_visualization_plot_2(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    clustering = PDDP(max_clusters_number=3).fit(data_import["data"])
    try:
        viz.split_visualization(clustering)
        assert True
    except Exception:
        assert False


def test_split_visualization_plot_3(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    clustering = BisectingKmeans(
        max_clusters_number=3,
    ).fit(data_import["data"])
    try:
        viz.split_visualization(clustering)
        assert True
    except Exception:
        assert False


def test_split_visualization_plot_4(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    clustering = KMPDDP(max_clusters_number=3).fit(data_import["data"])
    try:
        viz.split_visualization(clustering)
        assert True
    except Exception:
        assert False


def test_split_visualization_plot_5(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    clustering = IPDDP(max_clusters_number=3).fit(data_import["data"])
    try:
        viz.split_visualization(clustering)
        assert True
    except Exception:
        assert False


def test_split_visualization_plot_6(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    clustering = IPDDP(max_clusters_number=7).fit(data_import["data"])
    try:
        viz.split_visualization(clustering)
        assert True
    except Exception:
        assert False


def test_split_visualization_plot_7(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    clustering = IPDDP(max_clusters_number=2).fit(data_import["data"])
    try:
        viz.split_visualization(clustering)
        assert True
    except Exception:
        assert False


def test_split_visualization_plot_8(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    clustering = MDH(
        max_clusters_number=3,
        random_state=0,
    ).fit(data_import["data"])
    try:
        viz.split_visualization(clustering, mdh_split_plot=True)
        assert True
    except Exception:
        assert False


def test_split_visualization_plot_9(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    clustering = MDH(
        max_clusters_number=3,
        random_state=0,
    ).fit(data_import["data"])
    try:
        viz.split_visualization(clustering, mdh_split_plot=False)
        assert True
    except Exception:
        assert False


def test_split_visualization_typeerror(datadir):
    try:
        viz.split_visualization(np.array([1, 2, 3]))
        assert False
    except Exception:
        assert True


def test_split_visualization_valueerror_1(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    clustering = DePDDP(
        max_clusters_number=3,
        visualization_utility=False,
    ).fit(data_import["data"])

    try:
        viz.split_visualization(clustering)
        assert False
    except Exception:
        assert True


def test_split_visualization_valueerror_2(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    clustering = IPDDP(
        max_clusters_number=3,
        visualization_utility=False,
    ).fit(data_import["data"])

    try:
        viz.split_visualization(clustering)
        assert False
    except Exception:
        assert True


def test_dendrogram_visualization(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    clustering = DePDDP(max_clusters_number=3).fit(data_import["data"])
    new_plot = viz.dendrogram_visualization(clustering)

    assert isinstance(new_plot, dict)


def test_dendrogram_visualization_typeerror(datadir):
    try:
        viz.dendrogram_visualization(np.array([1, 2, 3]))
        assert False
    except Exception:
        assert True


def test_linkage(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    clustering = DePDDP(max_clusters_number=3).fit(data_import["data"])
    links = viz.linkage(clustering)

    assert isinstance(links, np.ndarray)


def test_linkage_typeerror(datadir):
    try:
        viz.linkage(np.array([1, 2, 3]))
        assert False
    except Exception:
        assert True


def test_utility_functions():
    success = 0

    try:
        uf.execute_decomposition_method(
            np.array([[1, 2, 3], [4, 5, 6]]),
            "tsne",
            True,
            {},
        )
    except Exception:
        success += 1
    try:
        uf.execute_decomposition_method(
            np.array([[1, 2, 3], [4, 5, 6]]),
            "asdf",
            False,
            {},
        )
    except Exception:
        success += 1

    assert success == 2
