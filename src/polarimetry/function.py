"""Helper functions for creating numerical functions from symbolic expressions."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Pattern

import jax.numpy as jnp

if TYPE_CHECKING:
    from tensorwaves.interface import DataSample, ParametrizedFunction

_LOGGER = logging.getLogger(__name__)


def compute_sub_function(
    func: ParametrizedFunction,
    input_data: DataSample,
    non_zero_couplings: list[str],
):
    old_parameters = dict(func.parameters)
    pattern = _get_coupling_regex(non_zero_couplings)
    set_parameter_to_zero(func, pattern)
    array = func(input_data)
    func.update_parameters(old_parameters)
    return array


def set_parameter_to_zero(
    func: ParametrizedFunction, search_term: str | Pattern[str]
) -> None:
    new_parameters = dict(func.parameters)
    no_parameters_selected = True
    for par_name in func.parameters:
        if re.match(search_term, par_name) is not None:
            new_parameters[par_name] = 0
            no_parameters_selected = False
    if no_parameters_selected:
        _LOGGER.warning(f"All couplings were set to zero for search term {search_term}")
    func.update_parameters(new_parameters)


def interference_intensity(func, data, chain1: list[str], chain2: list[str]) -> float:
    I_interference = sub_intensity(func, data, chain1 + chain2)
    I_chain1 = sub_intensity(func, data, chain1)
    I_chain2 = sub_intensity(func, data, chain2)
    return I_interference - I_chain1 - I_chain2


def sub_intensity(func, data, non_zero_couplings: list[str]):
    intensity_array = compute_sub_function(func, data, non_zero_couplings)
    return integrate_intensity(intensity_array)


def integrate_intensity(intensities) -> float:
    flattened_intensities = intensities.flatten()
    non_nan_intensities = flattened_intensities[~jnp.isnan(flattened_intensities)]
    return float(jnp.sum(non_nan_intensities) / len(non_nan_intensities))


def _get_coupling_regex(non_zero_couplings: list[str]) -> Pattern[str]:
    r"""Create regex pattern to match all couplings that should not be zero.

    >>> pat = _get_coupling_regex(["D", "K"])
    >>> print(pat)
    ^\\mathcal{H}\^\\mathrm{(LS,)?(decay|production)}\[(?!\\?(D|K)).*$
    >>> couplings = [
    ...     R"\mathcal{H}^\mathrm{decay}[\Delta(1232), -1/2, 0]",
    ...     R"\mathcal{H}^\mathrm{LS,production}[\Delta(1232), 1, 3/2]",
    ...     R"\mathcal{H}^\mathrm{decay}[K(700), 0, 0]",
    ...     R"\mathcal{H}^\mathrm{LS,production}[\Lambda(1405), 0, 1/2]",
    ... ]
    >>> [re.match(pat, coupling) is None for coupling in couplings]
    [True, True, True, False]
    """
    # https://regex101.com/r/BOGYEz
    non_zero_couplings = [re.escape(coupling) for coupling in non_zero_couplings]
    H = r"\\mathcal{H}\^\\mathrm{(LS,)?(decay|production)}"  # noqa: N806
    group = rf"({'|'.join(non_zero_couplings)})"
    return rf"^{H}\[(?!\\?{group}).*$"
