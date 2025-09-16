"""Make animations for particle trajectories from the output of OceanParcels simulations."""

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cartopy",
#     "cmocean",
#     "imageio-ffmpeg",
#     "matplotlib",
#     "numpy",
#     "tqdm",
#     "xarray[accel,io,parallel]",
# ]
# ///
import calendar
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean.cm as cmo
import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection
from numpy.typing import NDArray
from tqdm import tqdm

from overwrite_cli import parse_args

if TYPE_CHECKING:
    from cartopy.mpl.geoaxes import GeoAxes
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def open_parcels_output(
    path: str | Path,
    load: bool = True,  # noqa: FBT001,FBT002
) -> xr.Dataset:
    """Open a Zarr file containing OceanParcels (https://docs.oceanparcels.org/en/latest/index.html) output.

    Automatically detects if the path is a single zarr file or if it
    is a directory containing multiple Zarr files from MPI runs,
    as detailed here: https://docs.oceanparcels.org/en/latest/examples/documentation_MPI.html.

    Args:
        path (str or Path): Path to the Zarr file or directory containing the Zarr files.
        load (bool): If True, load the dataset into memory. Defaults to True.

    Returns:
        ds (xr.Dataset): The dataset containing the OceanParcels output.

    Raises:
        FileNotFoundError: If the specified path does not exist.

    """
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        msg = f"Path {path} does not exist."
        raise FileNotFoundError(msg)
    mpi_files = list(path.glob("proc*"))
    if len(mpi_files) == 0:
        ds = xr.open_zarr(path)
    else:
        ds = xr.concat(
            [xr.open_zarr(f) for f in mpi_files],
            dim="trajectory",
            compat="no_conflicts",
            coords="minimal",
        )
    if load:
        ds.load()  # Load the dataset into memory
    return ds


def make_animation(ds: xr.Dataset, save_path: Path | str) -> None:
    """Create an animation of the particle trajectories for a given year of particle releases.

    Args:
        ds (xr.Dataset): The dataset containing the particle trajectories.
        save_path (Path | str): The path where the animation will be saved.
        trajectory_colors (list | NDArray): An array of RGBA colors for each particle given by particle release month.

    """
    if isinstance(save_path, str):
        save_path = Path(save_path)

    fig: Figure
    ax: GeoAxes | Axes
    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={"projection": ccrs.PlateCarree()})
    fig.tight_layout(pad=2.0)
    # Need to explicitly cast to GeoAxes for type checking,
    # since plt.subplots doesn't return different types based on subplot_kw
    ax = cast("GeoAxes", ax)

    sca_data = np.array([ds["lon"].isel(obs=0), ds["lat"].isel(obs=0)])
    sca = ax.scatter(
        sca_data[0],
        sca_data[1],
        transform=ccrs.PlateCarree(),
        rasterized=True,
        s=1,
    )
    ax.coastlines()
    ax.set_extent([-131, -120, 40, 51], crs=ccrs.PlateCarree())
    ax.axhline(50, color="gray", linestyle="--", linewidth=1.5)

    gl = ax.gridlines(ls="--", color="gray", alpha=0.5, linewidth=0.5)
    gl.bottom_labels = True
    gl.left_labels = True
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle="-", linewidth=1.5)
    ax.add_feature(cfeature.STATES, linestyle=":")

    def init() -> tuple[PathCollection]:
        return (sca,)

    def update(frame: int, sca: PathCollection) -> tuple[PathCollection]:
        sca_data = np.array([ds["lon"].isel(obs=frame), ds["lat"].isel(obs=frame)])
        sca.set_offsets(sca_data.T)
        obs = ds["obs"].to_numpy()[frame]
        ax.set_title(obs)
        return (sca,)

    ani = FuncAnimation(fig, partial(update, sca=sca), frames=ds["obs"].size, init_func=init, blit=True, interval=50)

    ani.save(str(save_path), writer="ffmpeg")


def month_colors(year: int, n_particles_per_day: int) -> NDArray:
    """Generate a list of colors for each month in the given year.

    Args:
        year (int): The year for which to generate month colors.
        n_particles_per_day (int): Number of particles released per day.

    Returns:
        trajectory_colors (NDArray): An array of RGBA colors for each particle given by particle release month.

    """
    month_lengths = [calendar.monthrange(year, i)[1] for i in range(1, 13)]
    month_colors = [cmo.phase(i / 12) for i in range(1, 13)]  # type: ignore[reportAttributeAccessIssue]
    trajectory_colors = np.concatenate(
        [np.full((ml * n_particles_per_day, 4), color) for color, ml in zip(month_colors, month_lengths, strict=True)],
    )
    return trajectory_colors


if __name__ == "__main__":
    # Set the data directory and years to process
    DATA_DIR = Path("/mnt/d/plankton_particle_tracks/")
    years = [2005, 2008, 2011]
    depths = [5, 50]
    # Ensure the data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_dir = DATA_DIR / "animations"
    save_dir.mkdir(parents=True, exist_ok=True)
    args = parse_args("Make animations for particle trajectories from the output of OceanParcels simulations.")
    with tqdm(desc="Making animations", total=len(years) * len(depths)) as pbar:
        for year in years:
            for depth in depths:
                file = DATA_DIR / f"bwd_release_plankton_stations_{year}_{abs(depth)}m.zarr"
                save_path = save_dir / f"bwd_release_animation_{year}_{abs(depth)}m.mp4"

                if not file.exists():
                    print(f"File {file} does not exist. Skipping year {year}.")
                    pbar.update(1)
                    continue

                if save_path.exists() and not args.force:
                    if args.prompt:
                        response = input(f"Output file {save_path} already exists. Overwrite? (y/n): ")
                        if response.lower() != "y":
                            print(f"Skipping year {year}.")
                            pbar.update(1)
                            continue
                    else:
                        print(f"Skipping year {year}, output file already exists. Use --force to overwrite.")
                        pbar.update(1)
                        continue
                else:
                    with open_parcels_output(file) as ds:
                        make_animation(ds, save_path)
                        pbar.update(1)
