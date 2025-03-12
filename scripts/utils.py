import numpy as np
import xarray as xr


def make_landmask(path):
    landmask = xr.open_dataset(path).mask_rho_no_rivers.values
    landmask = np.invert(landmask.astype(bool)).astype(int)
    landmask = landmask.astype(int)
    return landmask


def get_coastal_nodes(landmask, shift=1):
    """Function that detects the coastal nodes, i.e. the ocean nodes directly
    next to land. Computes the Laplacian of landmask.

    - landmask: the land mask built using `make_landmask`, where land cell = 1
                and ocean cell = 0.

    Output: 2D array array containing the coastal nodes, the coastal nodes are
            equal to one, and the rest is zero.
    """
    mask_lap = np.roll(landmask, -shift, axis=0) + np.roll(landmask, shift, axis=0)
    mask_lap += np.roll(landmask, -shift, axis=1) + np.roll(landmask, shift, axis=1)
    mask_lap -= 4 * landmask
    coastal = np.ma.masked_array(landmask, mask_lap > 0)
    coastal = coastal.mask.astype("int")

    return coastal


def get_coastal_nodes_diagonal(landmask, shift=1):
    """Function that detects the coastal nodes, i.e. the ocean nodes where
    one of the 8 nearest nodes is land. Computes the Laplacian of landmask
    and the Laplacian of the 45 degree rotated landmask.

    - landmask: the land mask built using `make_landmask`, where land cell = 1
                and ocean cell = 0.

    Output: 2D array array containing the coastal nodes, the coastal nodes are
            equal to one, and the rest is zero.
    """
    mask_lap = np.roll(landmask, -shift, axis=0) + np.roll(landmask, shift, axis=0)
    mask_lap += np.roll(landmask, -shift, axis=1) + np.roll(landmask, shift, axis=1)
    mask_lap += np.roll(landmask, (-shift, shift), axis=(0, 1)) + np.roll(
        landmask, (shift, shift), axis=(0, 1)
    )
    mask_lap += np.roll(landmask, (-shift, -shift), axis=(0, 1)) + np.roll(
        landmask, (shift, -shift), axis=(0, 1)
    )
    mask_lap -= 8 * landmask
    coastal = np.ma.masked_array(landmask, mask_lap > 0)
    coastal = coastal.mask.astype("int")

    return coastal


def get_shore_nodes(landmask, shift=1):
    """Function that detects the shore nodes, i.e. the land nodes directly
    next to the ocean. Computes the Laplacian of landmask.

    - landmask: the land mask built using `make_landmask`, where land cell = 1
                and ocean cell = 0.

    Output: 2D array array containing the shore nodes, the shore nodes are
            equal to one, and the rest is zero.
    """
    mask_lap = np.roll(landmask, -shift, axis=0) + np.roll(landmask, shift, axis=0)
    mask_lap += np.roll(landmask, -shift, axis=1) + np.roll(landmask, shift, axis=1)
    mask_lap -= 4 * landmask
    shore = np.ma.masked_array(landmask, mask_lap < 0)
    shore = shore.mask.astype("int")

    return shore


def get_shore_nodes_diagonal(landmask, shift=1):
    """Function that detects the shore nodes, i.e. the land nodes where
    one of the 8 nearest nodes is ocean. Computes the Laplacian of landmask
    and the Laplacian of the 45 degree rotated landmask.

    - landmask: the land mask built using `make_landmask`, where land cell = 1
                and ocean cell = 0.

    Output: 2D array array containing the shore nodes, the shore nodes are
            equal to one, and the rest is zero.
    """
    mask_lap = np.roll(landmask, -shift, axis=0) + np.roll(landmask, shift, axis=0)
    mask_lap += np.roll(landmask, -shift, axis=1) + np.roll(landmask, shift, axis=1)
    mask_lap += np.roll(landmask, (-shift, shift), axis=(0, 1)) + np.roll(
        landmask, (shift, shift), axis=(0, 1)
    )
    mask_lap += np.roll(landmask, (-shift, -shift), axis=(0, 1)) + np.roll(
        landmask, (shift, -shift), axis=(0, 1)
    )
    mask_lap -= 8 * landmask
    shore = np.ma.masked_array(landmask, mask_lap < 0)
    shore = shore.mask.astype("int")

    return shore


def create_displacement_field(landmask, shift=1):
    """Function that creates a displacement field 1 m/s away from the shore.

    - landmask: the land mask dUilt using `make_landmask`.

    Output: two 2D arrays, one for each camponent of the velocity.
    """
    shore = get_coastal_nodes(landmask, shift)

    # nodes bordering ocean directly and diagonally
    shore_d = get_coastal_nodes_diagonal(landmask, shift)
    # corner nodes that only border ocean diagonally
    shore_c = shore_d - shore

    # Simple derivative
    Ly = np.roll(landmask, -shift, axis=0) - np.roll(landmask, shift, axis=0)
    Lx = np.roll(landmask, -shift, axis=1) - np.roll(landmask, shift, axis=1)

    Ly_c = np.roll(landmask, -shift, axis=0) - np.roll(landmask, shift, axis=0)
    # Include y-component of diagonal neighbors
    Ly_c += np.roll(landmask, (-shift, -shift), axis=(0, 1)) + np.roll(
        landmask, (-shift, shift), axis=(0, 1)
    )
    Ly_c += -np.roll(landmask, (shift, -shift), axis=(0, 1)) - np.roll(
        landmask, (shift, shift), axis=(0, 1)
    )

    Lx_c = np.roll(landmask, -shift, axis=1) - np.roll(landmask, shift, axis=1)
    # Include x-component of diagonal neighbors
    Lx_c += np.roll(landmask, (-shift, -shift), axis=(1, 0)) + np.roll(
        landmask, (-shift, shift), axis=(1, 0)
    )
    Lx_c += -np.roll(landmask, (shift, -shift), axis=(1, 0)) - np.roll(
        landmask, (shift, shift), axis=(1, 0)
    )

    v_x = -Lx * (shore)
    v_y = -Ly * (shore)

    v_x_c = -Lx_c * (shore_c)
    v_y_c = -Ly_c * (shore_c)

    v_x = v_x + v_x_c
    v_y = v_y + v_y_c

    magnitude = np.sqrt(v_y**2 + v_x**2)
    # the coastal nodes between land create a problem. Magnitude there is zero
    # I force it to be 1 to avoid problems when normalizing.
    ny, nx = np.where(magnitude == 0)
    magnitude[ny, nx] = 1

    v_x = v_x / magnitude
    v_y = v_y / magnitude

    return v_x, v_y


def distance_to_shore(landmask, dx=1, shift=1):
    """Function that computes the distance to the shore. It is based in the
    the `get_coastal_nodes` algorithm.

    - landmask: the land mask dUilt using `make_landmask` function.
    - dx: the grid cell dimension. This is a crude approxsimation of the real
    distance (be careful).

    Output: 2D array containing the distances from shore.
    """
    ci = get_coastal_nodes(landmask, shift)  # direct neighbors
    dist = ci * dx  # 1 dx away

    ci_d = get_coastal_nodes_diagonal(landmask, shift)  # diagonal neighbors
    dist_d = (ci_d - ci) * np.sqrt(2 * dx**2)  # sqrt(2) dx away

    return dist + dist_d
