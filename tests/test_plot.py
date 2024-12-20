import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.contour import QuadContourSet

from polarimetry.plot import stylize_contour


@pytest.fixture
def contour_set():
    x = np.linspace(-3.0, 3.0, 100)
    y = np.linspace(-3.0, 3.0, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sqrt(X**2 + Y**2)
    fig, ax = plt.subplots()
    contour_set = ax.contour(X, Y, Z)
    yield contour_set
    plt.close(fig)


def test_stylize_contour(contour_set: QuadContourSet):
    stylize_contour(contour_set, edgecolor="red")
    stylize_contour(contour_set, label="Test Label")
    stylize_contour(contour_set, linestyle="--")
    stylize_contour(contour_set, linewidth=2.0)
