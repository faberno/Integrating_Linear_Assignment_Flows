from euler import Euler
import numpy as np
import matplotlib.pyplot as plt


def plot_quiver(X: np.ndarray, width: float = None, grid: bool = False):
    """Plot 2D development of an ODE

    :param X:
    :param width:
    :param grid:
    :return:
    """
    min_bounds = X.min(axis=1)
    max_bounds = X.max(axis=1)
    dir = np.diff(X, axis=1)
    plt.quiver(X[0, :-1], X[1, :-1], dir[0], dir[1], angles='xy', scale_units='xy', scale=1, width=width)
    plt.xlim([min_bounds[0] * 1.1, max_bounds[0] * 1.1])
    plt.ylim([min_bounds[1] * 1.1, max_bounds[1] * 1.1])
    plt.gca().set_aspect('equal', adjustable='box')
    if grid:
        plt.grid()
    plt.show()
