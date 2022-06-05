from __future__ import annotations

import logging
import re
from typing import Pattern

from tensorwaves.function import ParametrizedBackendFunction


def compute_sub_function(
    func: ParametrizedBackendFunction, input_data, non_zero_couplings: list[str]
) -> None:
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
