import numpy as np
from colour.adaptation import chromatic_adaptation
from colour.colorimetry import CCS_ILLUMINANTS

def custom_CAT(XYZ, xy_wp_source, xy_wp_target, method = 'Von Kries'):
    """
    Apply custom Chromatic Adaptation Transform (CAT) to the given XYZ color values.

    Parameters:
    - XYZ: numpy array of shape (..., 3), the input color values in XYZ color space.
    - method: str, the CAT method to use (e.g., 'Bradford', 'Von Kries', etc.).
    - xy_wp_source: tuple, the xy chromaticity coordinates of the source white point.
    - xy_wp_target: tuple, the xy chromaticity coordinates of the target white point.

    Returns:
    - adapted_XYZ: numpy array of shape (..., 3), the adapted color values in XYZ color space.
    """
    # Convert xy chromaticity coordinates to XYZ
    def xy_to_XYZ(xy):
        x, y = xy
        Y = 1.0
        X = (x * Y) / y
        Z = ((1 - x - y) * Y) / y
        return np.array([X, Y, Z])

    XYZ_wp_source = xy_to_XYZ(xy_wp_source)
    XYZ_wp_target = xy_to_XYZ(xy_wp_target)

    # Apply chromatic adaptation
    adapted_XYZ = chromatic_adaptation(
        XYZ,
        XYZ_wp_source,
        XYZ_wp_target,
        method=method
    )

    return adapted_XYZ