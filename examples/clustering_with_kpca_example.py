import HiPart.visualizations as viz
import matplotlib.pyplot as plt
import numpy as np

from HiPart.clustering import dePDDP
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles


def plot_manafolds(X, y, vals, gamma, title):
    """
    2D plot for the generation of a scatter plot with the same meshgrid as the
    one used by the KernelPCA method.

    Parameters
    ----------
    X : numpy.ndarray
        Data for the scatter plot.
    y : numpy.ndarray
        Data labels.
    vals : list
        Label values.
    gamma : flaot
        Meshgrid gamma.
    title : str
        Title of the plot.

    Returns
    -------
    plt :
        The final plot.

    """
    plt.title(title)
    reds = y == vals[0]
    blues = y == vals[1]

    plt.scatter(X[reds, 0], X[reds, 1], c="red", s=20, edgecolor="k")
    plt.scatter(X[blues, 0], X[blues, 1], c="blue", s=20, edgecolor="k")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    X1, X2 = np.meshgrid(
        np.linspace(-gamma, gamma, 50),
        np.linspace(-gamma, gamma, 50),
    )
    X_grid = np.array([np.ravel(X1), np.ravel(X2)]).T
    # projection on the first principal component (in the phi space)
    Z_grid = kpca.transform(X_grid)[:, 0].reshape(X1.shape)
    plt.contour(X1, X2, Z_grid, colors="grey", linewidths=1, origin="lower")

    return plt

if __name__ == "__main__":
    # set a random seed for results consistency
    np.random.seed(123)

    # Create the dataset
    X, y = make_circles(n_samples=400, factor=0.3, noise=0.05)

    # execution of the for visulaization purposes
    kPCA_params = {
        "kernel": "rbf",
        "fit_inverse_transform": True,
        "gamma": 1.5,
    }
    kpca = KernelPCA(**kPCA_params)
    X_kpca = kpca.fit(X)

    # visualize the data on the original space
    plot_manafolds(
        X=X,
        y=y,
        vals=[0, 1],
        gamma=1.5,
        title="Original space",
    ).show()

    ##### data clustering

    clusternumber = 2

    outObj = dePDDP(
        decomposition_method="kpca",
        max_clusters_number=clusternumber,
        bandwidth_scale=0.5,
        percentile=0.1,
        **kPCA_params
    ).fit(X)

    #####

    # split visualization
    fig = viz.split_visualization(outObj)
    fig.show()

    # Clustered data visulization
    out_y = outObj.labels_
    plot_manafolds(
        X=X,
        y=out_y,
        vals=[1, 2],
        gamma=1.5,
        title="Clusterd data",
    ).show()
