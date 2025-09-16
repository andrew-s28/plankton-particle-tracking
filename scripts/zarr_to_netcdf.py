"""Converts all Zarr datasets in the current working directory to NetCDF files."""

from pathlib import Path

from tqdm import tqdm

from overwrite_cli import parse_args
from utils import open_parcels_output


def zarr_to_netcdf(
    zarr_path: str | Path,
    netcdf_path: str | Path,
) -> None:
    """Convert a Zarr dataset to a NetCDF file.

    Args:
        zarr_path (str or Path): Path to the input Zarr dataset.
        netcdf_path (str or Path): Path to the output NetCDF file.

    Raises:
        FileNotFoundError: If the specified Zarr path does not exist.

    """
    zarr_path = Path(zarr_path)

    if not zarr_path.exists():
        msg = f"Zarr path {zarr_path} does not exist."
        raise FileNotFoundError(msg)

    ds = open_parcels_output(zarr_path)
    ds.to_netcdf(netcdf_path)
    print(f"Converted {zarr_path} to {netcdf_path}")


if __name__ == "__main__":
    args = parse_args("Convert Zarr dataset to NetCDF format.")

    zarr_files = Path().cwd().glob("*.zarr")

    for zarr_file in tqdm(zarr_files, desc="Converting files"):
        netcdf_file = zarr_file.with_suffix(".nc")
        if netcdf_file.exists():
            if args.prompt:
                response = input(
                    f"NetCDF file {netcdf_file} already exists. Overwrite? (y/n): ",
                )
                if response.lower() != "y":
                    print("Skipping file:", netcdf_file)
                    continue
            elif not args.force:
                print(
                    f"NetCDF file {netcdf_file} already exists. Use --force to overwrite.",
                )
                continue
        zarr_to_netcdf(zarr_file, netcdf_file)
