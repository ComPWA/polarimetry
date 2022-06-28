from __future__ import annotations

import logging
import re
from typing import Pattern

import jax.numpy as jnp
from tensorwaves.function import ParametrizedBackendFunction
from tensorwaves.interface import DataSample


def compute_sub_function(
    func: ParametrizedBackendFunction,
    input_data: DataSample,
    non_zero_couplings: list[Pattern],
):
    old_parameters = dict(func.parameters)
    pattern = rf"\\mathcal{{H}}.*\[(?!{'|'.join(non_zero_couplings)})"
    set_parameter_to_zero(func, pattern)
    array = func(input_data)
    func.update_parameters(old_parameters)
    return array


def set_parameter_to_zero(
    func: ParametrizedBackendFunction, search_term: Pattern
) -> None:
    new_parameters = dict(func.parameters)
    no_parameters_selected = True
    for par_name in func.parameters:
        if re.match(search_term, par_name) is not None:
            new_parameters[par_name] = 0
            no_parameters_selected = False
    if no_parameters_selected:
        logging.warning(f"All couplings were set to zero for search term {search_term}")
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
