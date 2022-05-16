"""Helper functions for constructing `attrs` decorated classes."""
from __future__ import annotations

from collections import abc
from typing import TYPE_CHECKING, SupportsFloat

import sympy as sp
from attrs import Attribute

if TYPE_CHECKING:
    from polarization.decay import LSCoupling


def assert_spin_value(instance, attribute: Attribute, value: sp.Rational) -> None:
    if value.denominator not in {1, 2}:
        raise ValueError(
            f"{attribute.name} value should be integer or half-integer, not {value}"
        )


def to_ls(obj: LSCoupling | tuple[int, SupportsFloat] | None) -> LSCoupling:
    from polarization.decay import LSCoupling

    if obj is None:
        return None
    if isinstance(obj, LSCoupling):
        return obj
    if isinstance(obj, tuple):
        L, S = obj
        return LSCoupling(L, S)
    raise TypeError(f"Cannot convert {type(obj).__name__} to {LSCoupling.__name__}")


def to_range_tuple(
    obj: SupportsFloat | tuple[SupportsFloat, SupportsFloat]
) -> tuple[float, float]:
    if isinstance(obj, abc.Iterable):
        obj = tuple(obj)
        if len(obj) != 2:
            raise ValueError(f"Need two items for a range tuple, not {len(obj)}")
        return obj
    if isinstance(obj, SupportsFloat):
        return float(obj), float(obj)
    raise TypeError(f"Cannot convert {type(obj).__name__} to a range tuple")


def to_rational(obj: SupportsFloat) -> sp.Rational:
    return sp.Rational(obj)
