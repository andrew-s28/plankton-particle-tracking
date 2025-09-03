"""Wrapper for running OceanParcels simulations."""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "parcels",
#     "xarray[accel,io,parallel]",
#     "zarr<3",
# ]
# ///
from datetime import datetime, timedelta
from pathlib import Path

from overwrite_cli import parse_args
from plankton_tracking import ParcelsConfig, execute_backwards_release, write_metadata

local_tz = datetime.now().astimezone().tzinfo

if __name__ == "__main__":
    years = range(1980, 2017)
    depths = [-50]
    config = ParcelsConfig(
        d_lon_release=0.025,
        d_lat_release=0.025,
        depth_release=-50,
        n_particles=64,
        model_dir=Path("/mnt/d/avg"),
        input_dir=Path("/mnt/d/plankton_particle_tracks"),
        output_dir=Path("/mnt/d/plankton_particle_tracks/test"),
        runtime=timedelta(days=210),
        year_release=1997,
        max_age=timedelta(days=180),
    )
    args = parse_args()
    for year in years:
        for depth in depths:
            config.depth_release = depth
            config.update_year_release(year)
            if config.output_file.exists():
                if args.prompt:
                    response = input(
                        f"Output file for year {year} already exists. Overwrite? (y/n): ",
                    )
                    if response.lower() != "y":
                        print("Skipping year:", year)
                        continue
                elif not args.force:
                    print(
                        f"Output file for year {year} already exists. Use --force to overwrite.",
                    )
                    continue
            t_start = datetime.now(tz=local_tz)
            print(
                f"{t_start.strftime('%Y-%m-%d %H:%M:%S')}: "
                f"starting backward release for year {year} and depth {depth}.",
            )
            execute_backwards_release(config)
            write_metadata(config)
            t_finish = datetime.now(tz=local_tz)
            print(
                f"{t_finish.strftime('%Y-%m-%d %H:%M:%S')}: "
                f"completed backward release for year {year} and depth {depth} "
                f"in {timedelta(seconds=(t_finish - t_start).seconds)!s}.",
            )
