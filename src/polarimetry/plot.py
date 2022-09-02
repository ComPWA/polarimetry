"""Helper functions for `matplotlib`."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.contour import QuadContourSet


def get_contour_line(contour_set: QuadContourSet) -> LineCollection:
    (line_collection, *_), _ = contour_set.legend_elements()
    return line_collection


def use_mpl_latex_fonts(reset_mpl: bool = True) -> None:
    # cspell:ignore dejavusans fontset mathtext usetex
    if reset_mpl:
        _wake_up_matplotlib()
    plt.rc("font", family="serif", serif="Helvetica")
    plt.rc("mathtext", fontset="dejavusans")
    plt.rc("text", usetex=True)


def _wake_up_matplotlib() -> None:
    """Somehow `plt.rc` does not work if a figure hasn't been created before..."""
    plt.figure()
    plt.close()


def stylize_contour(
    contour_set: QuadContourSet,
    *,
    edgecolor=None,
    label: str | None = None,
    linestyle: str | None = None,
    linewidth: float | None = None,
) -> None:
    contour_line: PathCollection = contour_set.collections[0]
    if edgecolor is not None:
        contour_line.set_edgecolor(edgecolor)
    if label is not None:
        contour_line.set_label(label)
    if linestyle is not None:
        contour_line.set_linestyle(linestyle)
    if linewidth is not None:
        contour_line.set_linewidth(linewidth)