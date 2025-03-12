# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pandas",
#     "parcels",
#     "xarray",
#     "zarr<3",
# ]
# ///
import glob
import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

import parcels
from utils import (
    create_displacement_field,
    distance_to_shore,
    get_coastal_nodes_diagonal,
    get_shore_nodes_diagonal,
    make_landmask,
)

model_directory = "D:/avg/"
model_directory = os.path.abspath(model_directory)


landmask = make_landmask(os.path.join(model_directory, "croco_grd_no_rivers.nc.1b"))
coastal = get_coastal_nodes_diagonal(landmask)
shore = get_shore_nodes_diagonal(landmask)
u_displacement, v_displacement = create_displacement_field(landmask, shift=2)
d_2_s = distance_to_shore(landmask, shift=2)

station_file = os.path.join(os.path.dirname(__file__), "../data/StationPositions.csv")
stations = pd.read_csv(station_file)
n_particles = 50  # Number of particles to release per station
d_lat = 0.025  # Latitude range around stations for release
d_lon = 0.025  # Longitude range around stations for release
year = 2014  # Year of the release
runtime = timedelta(days=30)  # Total length of the run
dt = timedelta(minutes=-10)  # Timestep of the simulation
outputdt = timedelta(days=1)  # Output timestep

start_time = datetime(year, 7, 15)  # Start time of the release
end_time = datetime(year, 7, 14)  # End time of the release
output_path = f"../data/particle-tracks/bwd_release_plankton_stations_{year!s}_test.zarr"  # Output file name
output_chunks = (int(1e4), 100)  # Output chunk size
max_age = 365  # still need to change in the function (or use my hack :))

model_file_names = os.path.join(
    model_directory,
    f"{'[0-9]' * 4}/{'[0-9]' * 4}-complete.nc",
)
model_files = glob.glob(model_file_names)
model_files.sort()
grid = xr.open_dataset(os.path.join(model_directory, "croco_grd.nc.1b"))
# if os.path.exists(output_path):
#     raise FileExistsError(output_path)

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
chunksize = {
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
        "ignore", message="Multiple files given but no time dimension specified."
    )
    fieldset = parcels.FieldSet.from_croco(
        model_files,
        variables,
        dimensions,
        hc=200.0,
        deferred_load=True,
        chunksize=None,
    )
    fieldset.add_field(
        parcels.Field(
            "dispU",
            data=u_displacement,
            lon=grid.lon_rho,
            lat=grid.lat_rho,
        )
    )
    fieldset.add_field(
        parcels.Field(
            "dispV",
            data=u_displacement,
            lon=grid.lon_rho,
            lat=grid.lat_rho,
        )
    )
    fieldset.add_field(
        parcels.Field(
            "landmask",
            landmask,
            lon=grid.lon_rho,
            lat=grid.lat_rho,
        )
    )
    fieldset.add_field(
        parcels.Field(
            "distance2shore",
            d_2_s,
            lon=grid.lon_rho,
            lat=grid.lat_rho,
        )
    )

# Release time
release_start = start_time
release_end = end_time
runtime_release = release_end - release_start
release_time = xr.date_range(release_start, release_end, freq="-1D").values

n_particles = np.ceil(np.sqrt(n_particles)).astype(int)
shape = (release_time.size, stations.shape[0], n_particles**2)
re_lats = np.full(shape, np.nan)
re_lons = np.full(shape, np.nan)
re_depths = np.full(shape, np.nan)
for i, s in enumerate(stations.itertuples()):
    central_lon = s.lon
    central_lat = s.lat
    lons = np.linspace(central_lon - d_lon, central_lon + d_lon, n_particles)
    lats = np.linspace(central_lat - d_lat, central_lat + d_lat, n_particles)
    lons, lats = np.meshgrid(lons, lats)
    lons = lons.flatten()
    lats = lats.flatten()
    depths = np.full(n_particles**2, -5)
    re_lons[:, i] = np.tile(lons, (release_time.size, 1))
    re_lats[:, i] = np.tile(lats, (release_time.size, 1))
    re_depths[:, i] = np.tile(depths, (release_time.size, 1))

re_lons = re_lons.reshape(re_lons.shape[0], -1)
re_lats = re_lats.reshape(re_lats.shape[0], -1)
re_depths = re_depths.reshape(re_depths.shape[0], -1)
re_times = np.tile(release_time[:, None], (1, re_lons.shape[1]))
# print(fieldset.[0, 0, 44.645, -124.155])


class Particle(parcels.JITParticle):
    age = parcels.Variable("age", initial=-dt.total_seconds() / 86400)
    dU = parcels.Variable("dU", initial=0)
    dV = parcels.Variable("dV", initial=0)
    dlat = parcels.Variable("dlat", initial=0)
    dlon = parcels.Variable("dlon", initial=0)
    d2s = parcels.Variable("d2s", initial=0)
    landmask = parcels.Variable("landmask", initial=0)


def set_displacement(particle, fieldset, time):
    particle.d2s = fieldset.distance2shore[0, 0, particle.lat, particle.lon]
    if particle.d2s > 0:
        particle.dU = fieldset.dispU[0, 0, particle.lat, particle.lon]
        particle.dV = fieldset.dispV[0, 0, particle.lat, particle.lon]
        particle.dlon = particle.dU * particle.dt / 1e4
        particle.dlat = particle.dV * particle.dt / 1e4
        if (particle.dlon > 0) and (particle.dlat > 0):
            particle_dlon -= particle.dlon  # noqa: F821, F841
            particle_dlat -= particle.dlat  # noqa: F821, F841
    else:
        particle.dU = 0
        particle.dV = 0


def particle_age(particle, fieldset, time):
    if particle.time > 0:
        particle.age += particle.dt / 86400


def delete_old_particle(particle, fieldset, time):
    if particle.age > 365:
        particle.delete()
    if particle.age < -365:
        particle.delete()


def handle_error_particle(particle, fieldset, time):
    particle.landmask = fieldset.landmask[0, 0, particle.lat, particle.lon]
    if particle.landmask > 0:
        if particle.age > -1:
            particle.delete()
            # print("landmask is positive in handle_error_particle()")
            # print(particle.age)
    # these handle the rare case of particles being forced onto land by complicated coastline of Vancover Island
    if (particle.dlon < 0) and (particle.dlat < 0):
        particle.delete()
        # print("dlon and dlat are both positive in set_displacement()")
        # print(particle.age)
    if particle.state == StatusCode.ErrorOutOfBounds:  # type: ignore # noqa: F821
        if particle.age > -1:
            particle.delete()
            # print("ErrorOutOfBounds in handle_error_particle()")
            # print(particle.age)
        else:
            particle.dU += fieldset.dispU[0, 0, particle.lat, particle.lon]
            particle.dV += fieldset.dispV[0, 0, particle.lat, particle.lon]
            particle.dlon += particle.dU * particle.dt / 1e4
            particle.dlat += particle.dV * particle.dt / 1e4
            if (particle.dlon > 0) and (particle.dlat > 0):
                particle_dlon -= particle.dlon  # noqa: F821, F841
                particle_dlat -= particle.dlat  # noqa: F821, F841
            particle.state = StatusCode.Evaluate  # type: ignore # noqa: F821
    if particle.state == StatusCode.ErrorThroughSurface:  # type: ignore # noqa: F821
        particle_ddepth = 0  # noqa: F841
        particle.state = StatusCode.Success  # type: ignore # noqa: F821


kernels = [
    particle_age,
    set_displacement,
    parcels.AdvectionRK4_3D_CROCO,
    handle_error_particle,
    delete_old_particle,
]

pset = parcels.ParticleSet(
    fieldset=fieldset,
    pclass=Particle,
    lon=re_lons,
    lat=re_lats,
    depth=re_depths,
    time=re_times,
    # partition_function=partition_function,
)

output_file = pset.ParticleFile(
    name=output_path,
    outputdt=outputdt,
    chunks=output_chunks,
)

# with warnings.catch_warnings():
#     warnings.filterwarnings('ignore', message='The return type of `Dataset.dims` will be changed to return a set of dimension names in future, in order to be more consistent with `DataArray.dims`. To access a mapping from dimension names to lengths, please use `Dataset.sizes`.')
#     warnings.filterwarnings('ignore', message='The specified chunks separate the stored chunks along dimension "time" starting at index 1. This could degrade performance. Instead, consider rechunking after loading.')
#     warnings.filterwarnings('ignore', message='The specified chunks separate the stored chunks along dimension')
#     warnings.filterwarnings('ignore', message='Converting non-nanosecond precision datetime values')
#     warnings.filterwarnings('ignore', message='ParticleSet is empty on writing')
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", message="Multiple files given but no time dimension specified."
    )
try:
    pset.execute(
        kernels,
        runtime=runtime,
        dt=dt,
        output_file=output_file,
    )
except parcels.tools.statuscodes.TimeExtrapolationError as tee:
    print(tee)
    print(fieldset.U.grid.time)
    print(fieldset.gridset.dimrange("time"))

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# comm.barrier()

# try:
#     if rank == 0:
#         files = glob.glob(os.path.join(output_path, 'proc*'))
#         for f in files:
#             print(rank, f)
#             group = zarr.open_group(f)
#             group.attrs.update({
#                 'start_lat': str(start_lat),
#                 'end_lat': str(end_lat),
#                 'start_lon': str(start_lon),
#                 'end_lon': str(end_lon),
#                 'release_start_time': str(start_time),
#                 'release_end_time': str(end_time),
#                 'model_start_time': str(model_start_time),
#                 'model_end_time': str(model_end_time),
#                 'n_particles': str(n_particles),
#                 'partition_chunksize': str(partition_chunksize),
#                 'model_files_used': str(model_files),
#                 'runtime': str(runtime),
#                 'dt': str(dt),
#                 'outputdt': str(outputdt),
#                 'output_chunks': str(output_chunks),
#                 'max_age': str(max_age),
#             })
#             zarr.consolidate_metadata(f)
# except Exception as e:
#     print(e)
