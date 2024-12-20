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


def test_stylize_contour_edgecolor(contour_set: QuadContourSet):
    edgecolor = "red"
    stylize_contour(contour_set, edgecolor=edgecolor)


def test_stylize_contour_label(contour_set: QuadContourSet):
    label = "Test Label"
    stylize_contour(contour_set, label=label)


def test_stylize_contour_linestyle(contour_set: QuadContourSet):
    linestyle = "--"
    stylize_contour(contour_set, linestyle=linestyle)


def test_stylize_contour_linewidth(contour_set: QuadContourSet):
    linewidth = 2.0
    stylize_contour(contour_set, linewidth=linewidth)
