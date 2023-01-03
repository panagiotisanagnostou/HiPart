from __future__ import unicode_literals
from distutils import dir_util
from HiPart import visualizations as viz
from HiPart.clustering import dePDDP
from HiPart.clustering import iPDDP
from HiPart.clustering import kM_PDDP
from HiPart.clustering import PDDP
from HiPart.clustering import bicecting_kmeans

import numpy as np
import os
import pickle
import pytest


@pytest.fixture
def datadir(tmpdir, request):
    '''
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    '''
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir


def test_dePDDP_return_type(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    new_obj = dePDDP(max_clusters_number=5).fit(data_import["data"])
    assert isinstance(new_obj, dePDDP)


def test_iPDDP_return_type(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    new_obj = iPDDP(max_clusters_number=5).fit(data_import["data"])
    assert isinstance(new_obj, iPDDP)


def test_kM_PDDP_return_type(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    new_obj = kM_PDDP(max_clusters_number=5).fit(data_import["data"])
    assert isinstance(new_obj, kM_PDDP)


def test_PDDP_return_type(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    new_obj = PDDP(max_clusters_number=5).fit(data_import["data"])
    assert isinstance(new_obj, PDDP)


def test_bicecting_kmeans_return_type(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    new_obj = bicecting_kmeans(max_clusters_number=5).fit(data_import["data"])
    assert isinstance(new_obj, bicecting_kmeans)


def test_dePDDP_parameter_errors():
    success_score = 0

    algorithm = dePDDP()
    success_score += 1 if isinstance(algorithm.decomposition_method, str) else 0
    success_score += 1 if isinstance(algorithm.max_clusters_number, int) else 0
    success_score += 1 if isinstance(algorithm.split_data_bandwidth_scale, float) else 0
    success_score += 1 if isinstance(algorithm.percentile, float) else 0
    success_score += 1 if isinstance(algorithm.min_sample_split, int) else 0
    success_score += 1 if isinstance(algorithm.visualization_utility, bool) else 0

    try:
        dePDDP(decomposition_method="abc")
    except Exception:
        success_score += 1
    try:
        dePDDP(max_clusters_number=-5)
    except Exception:
        success_score += 1
    try:
        dePDDP(bandwidth_scale=-5)
    except Exception:
        success_score += 1
    try:
        dePDDP(percentile=.8)
    except Exception:
        success_score += 1
    try:
        dePDDP(min_sample_split=-5)
    except Exception:
        success_score += 1

    assert success_score == 11


def test_iPDDP_parameter_errors(datadir):
    success_score = 0

    algorithm = iPDDP()
    success_score += 1 if isinstance(algorithm.decomposition_method, str) else 0
    success_score += 1 if isinstance(algorithm.max_clusters_number, int) else 0
    success_score += 1 if isinstance(algorithm.percentile, float) else 0
    success_score += 1 if isinstance(algorithm.min_sample_split, int) else 0
    success_score += 1 if isinstance(algorithm.visualization_utility, bool) else 0

    try:
        iPDDP(decomposition_method="abc")
    except Exception:
        success_score += 1
    try:
        iPDDP(max_clusters_number=-5)
    except Exception:
        success_score += 1
    try:
        iPDDP(percentile=.8)
    except Exception:
        success_score += 1
    try:
        iPDDP(min_sample_split=-5)
    except Exception:
        success_score += 1

    assert success_score == 9


def test_kM_PDDP_parameter_errors(datadir):
    success_score = 0

    algorithm = kM_PDDP(random_seed=123)
    success_score += 1 if isinstance(algorithm.decomposition_method, str) else 0
    success_score += 1 if isinstance(algorithm.max_clusters_number, int) else 0
    success_score += 1 if isinstance(algorithm.min_sample_split, int) else 0
    success_score += 1 if isinstance(algorithm.random_seed, int) else 0
    success_score += 1 if isinstance(algorithm.visualization_utility, bool) else 0

    try:
        kM_PDDP(decomposition_method="abc")
    except Exception:
        success_score += 1
    try:
        kM_PDDP(max_clusters_number=-5)
    except Exception:
        success_score += 1
    try:
        kM_PDDP(random_seed=.8)
    except Exception:
        success_score += 1
    try:
        kM_PDDP(min_sample_split=-5)
    except Exception:
        success_score += 1

    assert success_score == 9


def test_PDDP_parameter_errors(datadir):
    success_score = 0

    algorithm = PDDP()
    success_score += 1 if isinstance(algorithm.decomposition_method, str) else 0
    success_score += 1 if isinstance(algorithm.max_clusters_number, int) else 0
    success_score += 1 if isinstance(algorithm.min_sample_split, int) else 0
    success_score += 1 if isinstance(algorithm.visualization_utility, bool) else 0

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

    assert success_score == 7


def test_bicecting_kmeans_parameter_errors():
    success_score = 0

    algorithm = bicecting_kmeans(random_seed=5)
    success_score += 1 if isinstance(algorithm.max_clusters_number, int) else 0
    success_score += 1 if isinstance(algorithm.min_sample_split, int) else 0
    success_score += 1 if isinstance(algorithm.random_seed, int) else 0

    try:
        bicecting_kmeans(max_clusters_number=-5)
    except Exception:
        success_score += 1
    try:
        bicecting_kmeans(random_seed=.8)
    except Exception:
        success_score += 1
    try:
        bicecting_kmeans(min_sample_split=-5)
    except Exception:
        success_score += 1

    assert success_score == 6


def test_dePDDP_labels__return_type_and_form(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    results = dePDDP(max_clusters_number=5).fit_predict(data_import["data"])
    assert isinstance(results, np.ndarray) and results.ndim == 1


def test_iPDDP_labels__return_type_and_form(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    results = iPDDP(max_clusters_number=5).fit_predict(data_import["data"])
    assert isinstance(results, np.ndarray) and results.ndim == 1


def test_kM_PDDP_labels__return_type_and_form(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    results = kM_PDDP(max_clusters_number=5).fit_predict(data_import["data"])
    assert isinstance(results, np.ndarray) and results.ndim == 1


def test_PDDP_labels__return_type_and_form(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    results = PDDP(max_clusters_number=5).fit_predict(data_import["data"])
    assert isinstance(results, np.ndarray) and results.ndim == 1


def test_bicecting_kmeans_labels__return_type_and_form(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    results = bicecting_kmeans(
        max_clusters_number=5,
    ).fit_predict(data_import["data"])
    assert isinstance(results, np.ndarray) and results.ndim == 1


def test_bicecting_kmeans_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["bicecting_kmeans"]

    matrix_test = bicecting_kmeans(
        max_clusters_number=5,
        random_seed=1256,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 6000


def test_dePDDP_pca_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["dePDDP_pca"]

    matrix_test = dePDDP(
        max_clusters_number=5,
        random_state=1256,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 6000


def test_iPDDP_pca_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["iPDDP_pca"]

    matrix_test = iPDDP(
        max_clusters_number=5,
        random_state=1256,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 6000


def test_kM_PDDP_pca_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["kM_PDDP_pca"]

    matrix_test = kM_PDDP(
        max_clusters_number=5,
        random_seed=1256,
        random_state=1256,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 6000


def test_PDDP_pca_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["PDDP_pca"]

    matrix_test = PDDP(
        max_clusters_number=5,
        random_state=1256,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 6000


def test_dePDDP_ica_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["dePDDP_ica"]

    matrix_test = dePDDP(
        decomposition_method="ica",
        max_clusters_number=5,
        random_state=1256,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 6000


def test_iPDDP_ica_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["iPDDP_ica"]

    matrix_test = iPDDP(
        decomposition_method="ica",
        max_clusters_number=5,
        random_state=1256,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 6000


def test_kM_PDDP_ica_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["kM_PDDP_ica"]

    matrix_test = kM_PDDP(
        decomposition_method="ica",
        max_clusters_number=5,
        random_seed=1256,
        random_state=1256,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 6000


def test_PDDP_ica_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["PDDP_ica"]

    matrix_test = PDDP(
        decomposition_method="ica",
        max_clusters_number=5,
        random_state=1256,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 6000


def test_dePDDP_kpca_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["dePDDP_kpca"]

    matrix_test = dePDDP(
        decomposition_method="kpca",
        max_clusters_number=5,
        random_state=1256,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 6000


def test_iPDDP_kpca_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["iPDDP_kpca"]

    matrix_test = iPDDP(
        decomposition_method="kpca",
        max_clusters_number=5,
        random_state=1256,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 6000


def test_kM_PDDP_kpca_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["kM_PDDP_kpca"]

    matrix_test = kM_PDDP(
        decomposition_method="kpca",
        max_clusters_number=5,
        random_seed=1256,
        random_state=1256,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 6000


def test_PDDP_kpca_results(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    matrix_control = data_import["PDDP_kpca"]

    matrix_test = PDDP(
        decomposition_method="kpca",
        max_clusters_number=5,
        random_state=1256,
    ).fit(data_import["data"]).output_matrix
    assert np.sum(matrix_test == matrix_control) == 6000


def test_split_visualization_plot_1(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    clustering = dePDDP(max_clusters_number=5).fit(data_import["data"])
    try:
        viz.split_visualization(clustering)
        assert True
    except Exception:
        assert False


def test_split_visualization_plot_2(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    clustering = PDDP(max_clusters_number=5).fit(data_import["data"])
    try:
        viz.split_visualization(clustering)
        assert True
    except Exception:
        assert False


def test_split_visualization_plot_3(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    clustering = bicecting_kmeans(
        max_clusters_number=5,
    ).fit(data_import["data"])
    try:
        viz.split_visualization(clustering)
        assert True
    except Exception:
        assert False


def test_split_visualization_plot_4(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    clustering = kM_PDDP(max_clusters_number=5).fit(data_import["data"])
    try:
        viz.split_visualization(clustering)
        assert True
    except Exception:
        assert False


def test_split_visualization_plot_5(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    clustering = iPDDP(max_clusters_number=5).fit(data_import["data"])
    try:
        viz.split_visualization(clustering)
        assert True
    except Exception:
        assert False


def test_split_visualization_plot_6(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    clustering = iPDDP(max_clusters_number=7).fit(data_import["data"])
    try:
        viz.split_visualization(clustering)
        assert True
    except Exception:
        assert False


def test_split_visualization_plot_7(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    clustering = iPDDP(max_clusters_number=2).fit(data_import["data"])
    try:
        viz.split_visualization(clustering)
        assert True
    except Exception:
        assert False


def test_split_visualization_TypeError(datadir):

    try:
        viz.split_visualization(np.array([1, 2, 3]))
        assert False
    except Exception:
        assert True


def test_split_visualization_ValueError_1(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    clustering = dePDDP(
        max_clusters_number=5,
        visualization_utility=False,
    ).fit(data_import["data"])

    try:
        viz.split_visualization(clustering)
        assert False
    except Exception:
        assert True


def test_split_visualization_ValueError_2(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    clustering = iPDDP(
        max_clusters_number=5,
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

    clustering = dePDDP(max_clusters_number=5).fit(data_import["data"])
    new_plot = viz.dendrogram_visualization(clustering)

    assert isinstance(new_plot, dict)


def test_dendrogram_visualization_TypeError(datadir):
    try:
        viz.dendrogram_visualization(np.array([1, 2, 3]))
        assert False
    except Exception:
        assert True


def test_linkage(datadir):
    with open(datadir.join('test_data.dump'), "rb") as inf:
        data_import = pickle.load(inf)

    clustering = dePDDP(max_clusters_number=5).fit(data_import["data"])
    links = viz.linkage(clustering)

    assert isinstance(links, np.ndarray)


def test_linkage_TypeError(datadir):
    try:
        viz.linkage(np.array([1, 2, 3]))
        assert False
    except Exception:
        assert True
