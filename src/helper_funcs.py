"""
functions that are helpful to both the creation
of pump dataframes, and plotting
"""
# pylint: disable=C0103
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np  # type: ignore
import numpy.polynomial.polynomial as poly  # type: ignore
import yaml  # type: ignore


def read_config_file(config_file_dir: Union[str, Path]) -> Dict:
    """
    parsed yml config file into dict
    """
    with open(config_file_dir, "r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file)
    return config_dict


def poly_fit(
    x: Sequence, y: Sequence, deg: int = 2, x_new: Optional[Iterable] = None
) -> np.ndarray:
    """
    if x_new is not provided:
        returns the coefficients of the best fit line corresponding
        to a fit of degree *def* over data *x* and *y*
    If x_new is not None:
        returns a predicted y values for x_new given the best fit curve calculated
    """
    if x_new is None:
        return poly.polyfit(x, y, deg)
    coefs = poly.polyfit(x, y, deg)
    return poly.polyval(x_new, coefs)


def plot_colors_dict() -> dict:
    """
    func to return colors for plotting, courtesy of https://htmlcolorcodes.com/
    """
    colors_dict = {
        "yellow": "#F1C40F",
        "blue": "#21618C",
        "red": "#CB4335",
        "green": "#1E8449",
        "orange": "#AF601A",
        "purple": "#6C3483",
        "black": "#333333",

    }
    return colors_dict
