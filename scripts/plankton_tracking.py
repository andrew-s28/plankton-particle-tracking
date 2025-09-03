"""Primary code for setting up and executing an OceanParcels Lagrangian simulation."""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pandas",
#     "parcels",
#     "xarray[io,accel,parallel]",
#     "zarr<3",
# ]
# ///

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

import parcels
from utils import (
    compute_distance_to_shore,
    create_displacement_field,
    make_landmask,
)

print(f"Parcels version: {parcels.__version__}")


@dataclass
class ParcelsConfig:
    """Dataclass to hold the configuration for Parcels simulations.

    Attributes:
        d_lon_release (float): Longitude range for particle release around each station.
        d_lat_release (float): Latitude range for particle release around each station.
        depth_release (float): Depth at which particles are released.
        n_particles (int): Number of particles to release at each station.
        model_dir (Path): Directory containing the primary model and grid files.
        input_dir (Path): Directory for additional input files if needed.
        output_dir (Path): Directory where output files will be saved.
        runtime (timedelta): Total runtime of the simulation.
        year_release (int): Year of release for the particles.
        distance_to_shore_limit (float): Distance to shore limit for offshore displacement, between 0 and 1.
        output_chunks (tuple[int, int]): Output chunk size for the particle file.
        output_timestep (timedelta): Timestep for the particle file output.
        advection_timestep (timedelta): Timestep of the advection simulation.
        max_age (timedelta): Maximum age of particles before deletion.

    Raises:
        FileNotFoundError: If no model files are found for the specified year.

    """

    d_lon_release: float
    d_lat_release: float
    depth_release: float
    n_particles: int
    model_dir: Path  # primary model and grid files directory
    input_dir: Path  # for additional input files if needed
    output_dir: Path
    runtime: timedelta
    year_release: int
    distance_to_shore_limit: float = 0.1  # Distance to shore limit for offshore displacement, between 0 and 1
    output_chunks: tuple[int, int] = (10_000, 100)  # Output chunk size
    output_timestep: timedelta = timedelta(
        days=1,
    )  # Output timestep for the particle file
    advection_timestep: timedelta = timedelta(
        minutes=-10,
    )  # Timestep of the advection simulation
    max_age: timedelta = timedelta(days=365)  # Maximum age of particles before deletion

    def __post_init__(self) -> None:
        """Initialize the ParcelsConfig dataclass."""
        self.set_start_end_times()
        self.set_output_file()
        self.set_model_files()
        self.set_station_file()

    def update_year_release(self, year_release: int) -> None:
        """Update the year of release and reconfigure the model files, start and end times, and output file."""
        self.year_release = year_release
        self.set_start_end_times()
        self.set_output_file()
        self.set_model_files()

    def set_model_files(self) -> None:
        """Set the model files based on the year of release.

        Raises:
            FileNotFoundError: If no model files are found for the specified year.

        """
        model_file_years = np.arange(self.year_release - 1, self.year_release + 2)
        # Get the model files for the three years around the release year
        self.model_files = [self.model_dir / f"{year:04d}/{year:04d}-complete.nc" for year in model_file_years]
        self.model_files.sort()
        if not self.model_files:
            msg = f"No model files found for year {self.year_release} in {self.model_dir}."
            raise FileNotFoundError(msg)

    def set_station_file(self) -> None:
        """Set the station files based on the year of release."""
        self.station_file = self.input_dir / "StationPositionsMidshelf.csv"

    def set_start_end_times(self) -> None:
        """Set the start and end times for the simulation."""
        self.start_time = datetime(self.year_release, 7, 15, tzinfo=UTC)
        self.end_time = datetime(self.year_release, 6, 15, tzinfo=UTC)

    def set_output_file(self) -> None:
        """Set the output file path based on the year."""
        if self.start_time > self.end_time:
            self.output_file = (
                self.output_dir / f"bwd_release_plankton_stations_{self.year_release!s}_{abs(self.depth_release)}m.zarr"
            )
        else:
            self.output_file = (
                self.output_dir / f"fwd_release_plankton_stations_{self.year_release!s}_{abs(self.depth_release)}m.zarr"
            )


def get_release_positions(  # noqa: PLR0913
    station_file: Path,
    start_time: datetime,
    end_time: datetime,
    n_particles: int,
    d_lon_release: float,
    d_lat_release: float,
    depth_release: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the release positions for the particles based on the station file.

    Args:
        station_file (Path): Path to the CSV file containing station coordinates.
        start_time (datetime): Start time of the simulation.
        end_time (datetime): End time of the simulation.
        n_particles (int): Number of particles to release at each station.
        d_lon_release (float): Longitude range for particle release around each station.
        d_lat_release (float): Latitude range for particle release around each station.
        depth_release (float): Depth at which particles are released.

    Returns:
        (release_lons, release_lats, release_depths, release_times) (tuple):
            A tuple containing arrays of release longitudes, latitudes, depths, and times.

    Raises:
        TypeError: If the station coordinates are not valid floats.

    """
    # Setup release locations and times
    stations = pd.read_csv(station_file)
    release_times = xr.date_range(start_time, end_time, freq="-1D").to_numpy(
        dtype="datetime64[ns]",
    )
    # Round up the number of particles to the nearest perfect square
    n_particles = np.ceil(np.sqrt(n_particles)).astype(int)
    shape = (release_times.size, stations.shape[0], n_particles**2)
    release_lons = np.full(shape, np.nan)
    release_lats = np.full(shape, np.nan)
    release_depths = np.full(shape, np.nan)
    for i, s in enumerate(stations.itertuples()):
        central_lon = s.lon
        central_lat = s.lat
        if isinstance(central_lon, int | float) and isinstance(
            central_lat,
            int | float,
        ):
            lons = np.linspace(
                central_lon - d_lon_release,
                central_lon + d_lon_release,
                n_particles,
            )
            lats = np.linspace(
                central_lat - d_lat_release,
                central_lat + d_lat_release,
                n_particles,
            )
            lons, lats = np.meshgrid(lons, lats)
            lons = lons.flatten()
            lats = lats.flatten()
            depths = np.full(n_particles**2, depth_release)
            release_lons[:, i] = np.tile(lons, (release_times.size, 1))
            release_lats[:, i] = np.tile(lats, (release_times.size, 1))
            release_depths[:, i] = np.tile(depths, (release_times.size, 1))
        else:
            msg = f"Invalid station coordinates for {s.name}: lon={central_lon}, lat={central_lat}. "
            msg += "Ensure that the station coordinates are valid floats."
            raise TypeError(msg)

    release_lons = release_lons.reshape(release_lons.shape[0], -1)
    release_lats = release_lats.reshape(release_lats.shape[0], -1)
    release_depths = release_depths.reshape(release_depths.shape[0], -1)
    release_times = np.tile(release_times[:, None], (1, release_lons.shape[1]))

    return release_lons, release_lats, release_depths, release_times


def get_landmask(
    model_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the landmask from the model grid file.

    Args:
        model_dir (Path): Directory containing the model grid file.

    Returns:
        np.ndarray: Landmask array where land is 1 and water is 0.

    """
    grid = xr.open_dataset(model_dir / "croco_grd.nc.1b")

    landmask = make_landmask(model_dir / "croco_grd_no_rivers.nc.1b")
    u_displacement, v_displacement = create_displacement_field(landmask, shift=2)
    distance_to_shore = compute_distance_to_shore(landmask, shift=2)

    # Filter the landmask and displacement fields to remove coastline
    # of Vancouver Island and bottom of model domain from the displacement kernels
    min_coast_lon = -125
    min_coast_lat = 30.5
    landmask = np.where(
        (grid["lon_rho"] > min_coast_lon) & (grid["lat_rho"] > min_coast_lat),
        landmask,
        0,
    )
    u_displacement = np.where(
        (grid["lon_rho"] > min_coast_lon) & (grid["lat_rho"] > min_coast_lat),
        u_displacement,
        0,
    )
    v_displacement = np.where(
        (grid["lon_rho"] > min_coast_lon) & (grid["lat_rho"] > min_coast_lat),
        v_displacement,
        0,
    )
    distance_to_shore = np.where(
        (grid["lon_rho"] > min_coast_lon) & (grid["lat_rho"] > min_coast_lat),
        distance_to_shore,
        0,
    )

    return landmask, u_displacement, v_displacement, distance_to_shore


def setup_fieldset(config: ParcelsConfig) -> parcels.FieldSet:
    """Create the fieldset for particle tracking.

    Args:
        config (ParcelsConfig): Configuration object containing model and grid information.

    Returns:
        fieldset (parcels.FieldSet): FieldSet object containing the necessary fields for particle tracking.

    """
    grid = xr.open_dataset(config.model_dir / "croco_grd.nc.1b")

    # Get the landmask and displacement fields
    landmask, u_displacement, v_displacement, distance_to_shore = get_landmask(
        config.model_dir,
    )

    variables = {
        "U": "u",
        "V": "v",
        "W": "w",
        "H": "h",
        "temp": "temp",
        "salt": "salt",
        "Zeta": "zeta",
        "Cs_w": "Cs_w",
    }
    dimensions = {
        "U": {"lon": "lon_rho", "lat": "lat_rho", "depth": "s_w", "time": "time"},
        "V": {"lon": "lon_rho", "lat": "lat_rho", "depth": "s_w", "time": "time"},
        "W": {"lon": "lon_rho", "lat": "lat_rho", "depth": "s_w", "time": "time"},
        "H": {"lon": "lon_rho", "lat": "lat_rho"},
        "temp": {"lon": "lon_rho", "lat": "lat_rho", "depth": "s_w", "time": "time"},
        "salt": {"lon": "lon_rho", "lat": "lat_rho", "depth": "s_w", "time": "time"},
        "Zeta": {"lon": "lon_rho", "lat": "lat_rho", "time": "time"},
        "Cs_w": {"depth": "s_w"},
    }
    chunksize = {  # noqa: F841 - not used currently, kept for reference
        "U": {
            "time": ("time", 1),
            "s_w": ("depth", 1),
            "lat_rho": ("lat", 1),
            "lon_rho": ("lon", 1),
        },
        "V": {
            "time": ("time", 1),
            "s_w": ("depth", 1),
            "lat_rho": ("lat", 1),
            "lon_rho": ("lon", 1),
        },
        "W": {
            "time": ("time", 1),
            "s_w": ("depth", 1),
            "lat_rho": ("lat", 1),
            "lon_rho": ("lon", 1),
        },
        "H": {"lat_rho": ("lat", 1), "lon_rho": ("lon", 1)},
        "temp": {
            "time": ("time", 1),
            "s_w": ("depth", 1),
            "lat_rho": ("lat", 1),
            "lon_rho": ("lon", 1),
        },
        "salt": {
            "time": ("time", 1),
            "s_w": ("depth", 1),
            "lat_rho": ("lat", 1),
            "lon_rho": ("lon", 1),
        },
        "Zeta": {"time": ("time", 1), "lat_rho": ("lat", 1), "lon_rho": ("lon", 1)},
        "Cs_w": {"s_w": ("depth", 1)},
    }

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=parcels.FieldSetWarning,
            message="Multiple files given but no time dimension specified.",
        )
        fieldset = parcels.FieldSet.from_croco(
            config.model_files,
            variables,
            dimensions,
            hc=200.0,
            deferred_load=True,
            chunksize=None,
        )

    fieldset.add_field(
        parcels.Field(
            "disp_u",
            data=u_displacement,
            lon=grid.lon_rho,
            lat=grid.lat_rho,
        ),
    )
    fieldset.add_field(
        parcels.Field(
            "disp_v",
            data=v_displacement,
            lon=grid.lon_rho,
            lat=grid.lat_rho,
        ),
    )
    fieldset.add_field(
        parcels.Field(
            "landmask",
            landmask,
            lon=grid.lon_rho,
            lat=grid.lat_rho,
        ),
    )
    fieldset.add_field(
        parcels.Field(
            "distance_to_shore",
            distance_to_shore,
            lon=grid.lon_rho,
            lat=grid.lat_rho,
        ),
    )
    fieldset.add_constant(
        "max_age",
        config.runtime.total_seconds() if config.max_age is None else config.max_age.total_seconds(),
    )
    fieldset.add_constant(
        "distance_to_shore_limit",
        config.distance_to_shore_limit,
    )  # Distance to shore limit for offshore displacement

    return fieldset


def setup_kernels() -> list[Callable[..., None]]:  # noqa: C901
    """Create the kernels for particle tracking.

    Returns:
        kernels (list[Callable[..., None]]): The list of kernel functions.

    """

    def set_displacement(particle, fieldset, time):  # noqa: ARG001, ANN001, ANN202
        particle.distance_to_shore = fieldset.distance_to_shore[
            0,
            0,
            particle.lat,
            particle.lon,
        ]
        particle.dlon = 0
        particle.dlat = 0
        particle.du = 0
        particle.dv = 0
        # If particle is close to land, manually add offshore displacement
        if particle.distance_to_shore > fieldset.distance_to_shore_limit:
            particle.du = fieldset.disp_u[0, 0, particle.lat, particle.lon]
            particle.dv = fieldset.disp_v[0, 0, particle.lat, particle.lon]
            particle.dlon = particle.du * particle.dt / 1e3
            particle.dlat = particle.dv * particle.dt / 1e3
            if (particle.dlon > 0) and (particle.dlat > 0):
                particle_dlon -= particle.dlon  # pyright: ignore[reportUnboundVariable]  # noqa: F821, F841
                particle_dlat -= particle.dlat  # pyright: ignore[reportUnboundVariable]  # noqa: F821, F841

    def particle_age(particle, fieldset, time):  # noqa: ARG001, ANN001, ANN202
        particle.age += particle.dt

    def delete_old_particle(particle, fieldset, time):  # noqa: ARG001, ANN001, ANN202
        # Handle forward releases
        if particle.age > fieldset.max_age:
            particle.delete()
            particle.delete_by_age = 1
        # Handle backward releases
        if particle.age < -fieldset.max_age:
            particle.delete()
            particle.delete_by_age = 1
        if particle.age >= -fieldset.max_age and particle.age <= fieldset.max_age:
            particle.delete_by_age = 0

    def handle_error_particle(particle, fieldset, time):  # noqa: ARG001, ANN001, ANN202
        particle.landmask = fieldset.landmask[0, 0, particle.lat, particle.lon]
        # If the manual dlon and dlat are negative, the particle is being pushed more onto land, so just delete it
        if (particle.dlon < 0) and (particle.dlat < 0):
            particle.delete()
            particle.delete_by_distance_to_shore = 1
        else:
            particle.delete_by_distance_to_shore = 0
        if particle.state == StatusCode.ErrorOutOfBounds:  # pyright: ignore[reportUndefinedVariable]  # noqa: F821
            particle.delete()
            particle.delete_by_out_of_bound = 1
        else:
            particle.delete_by_out_of_bound = 0
        if particle.state == StatusCode.ErrorThroughSurface:  # pyright: ignore[reportUndefinedVariable]  # noqa: F821
            particle_ddepth = 0  # noqa: F841
            particle.state = StatusCode.Success  # pyright: ignore[reportUndefinedVariable]  # noqa: F821

    def sample_fields(particle, fieldset, time):  # noqa: ANN001, ANN202
        particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]
        particle.salt = fieldset.salt[time, particle.depth, particle.lat, particle.lon]
        particle.h = fieldset.H[time, 0, particle.lat, particle.lon]

    kernels = [
        particle_age,
        set_displacement,
        parcels.AdvectionRK4_3D_CROCO,
        sample_fields,
        handle_error_particle,
        delete_old_particle,
    ]

    return kernels


def execute_backwards_release(config: ParcelsConfig) -> None:
    """Execute the backward release simulation using Parcels.

    Args:
        config (ParcelsConfig): Configuration object containing simulation parameters.

    """
    # Setup the fieldset
    fieldset = setup_fieldset(config)

    # Setup the kernels
    kernels = setup_kernels()

    # Get the release positions
    release_lons, release_lats, release_depths, release_times = get_release_positions(
        config.station_file,
        config.start_time,
        config.end_time,
        config.n_particles,
        config.d_lon_release,
        config.d_lat_release,
        config.depth_release,
    )

    class Particle(parcels.JITParticle):
        age = parcels.Variable(
            "age",
            initial=int(-config.advection_timestep.total_seconds()),
        )
        h = parcels.Variable("h", initial=int(config.depth_release))
        temp = parcels.Variable("temp", initial=0)
        salinity = parcels.Variable("salt", initial=0)
        du = parcels.Variable("du", initial=0)
        dv = parcels.Variable("dv", initial=0)
        dlat = parcels.Variable("dlat", initial=0)
        dlon = parcels.Variable("dlon", initial=0)
        distance_to_shore = parcels.Variable("distance_to_shore", initial=0)
        landmask = parcels.Variable("landmask", initial=0)
        delete_by_age = parcels.Variable("delete_by_age", initial=0)
        delete_by_distance_to_shore = parcels.Variable(
            "delete_by_distance_to_shore",
            initial=0,
        )
        delete_by_out_of_bound = parcels.Variable("delete_by_out_of_bound", initial=0)

    pset = parcels.ParticleSet(
        fieldset=fieldset,
        pclass=Particle,
        lon=release_lons,
        lat=release_lats,
        depth=release_depths,
        time=release_times,
    )

    output_particle_file = pset.ParticleFile(
        name=config.output_file,
        outputdt=config.output_timestep,
        chunks=config.output_chunks,
    )

    pset.execute(
        kernels,
        runtime=config.runtime,
        dt=config.advection_timestep,
        output_file=output_particle_file,
    )


def write_metadata(config: ParcelsConfig) -> None:
    """Write metadata to the output file."""
    metadata = {
        "d_lon": str(config.d_lon_release),
        "d_lat": str(config.d_lat_release),
        "min_time": str(config.start_time),
        "max_time": str(config.end_time),
        "n_particles": str(config.n_particles),
        "input_velocity_fields": str(config.model_files),
        "runtime": str(config.runtime),
        "advection_timestep": str(config.advection_timestep),
        "output_timestep": str(config.output_timestep),
        "output_chunks": str(config.output_chunks),
    }

    group = zarr.open_group(config.output_file)
    group.attrs.update(
        metadata,
    )
    zarr.consolidate_metadata(config.output_file)  # type: ignore[reportArgumentType]
