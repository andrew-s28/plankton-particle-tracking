# Scientific Analysis Code Template

This is the analysis code and datasets used for running Lagrangian particle tracking simulations in the Northern California Current System. 

## Usage

The primary Lagrangian particle tracking simulations are ran using [plankton_tracking.py](scripts/plankton_tracking.py). Run this script by:

1. `git clone https://github.com/andrew-s28/plankton-particle-tracking.git`

2. `uv run --script scripts/plankton_tracking.py`

      :warning: The [local install](scripts/parcels/) of [OceanParcels](https://github.com/OceanParcels/Parcels) is necessary to run this notebook since the fix implemented in [#1886](https://github.com/OceanParcels/Parcels/pull/1886) is not yet published on PyPi. If or when it is, this local install can be removed. Also, 
      :warning: A C compiler is necessary to run this script. See below for install instructions.

## Installing a C Code Compiler

OceanParcels requires a C compiler (such as [GCC](https://gcc.gnu.org/)) in the path to run - i.e., `gcc --version` should successfully run from the command line. If using a Linux distro (or even WSL), this is straightforward: enjoy the ease with which `sudo apt install build-essential` handles things perfectly!

On Windows, however, I find the install of GCC to be a bit more of a pain. To simplify, I use `conda` to install GCC into the base environment using the [m2w64-toolchain](https://anaconda.org/conda-forge/m2w64-toolchain) package and then add the library to the path. Something like this (for PowerShell) should get things up and running:

- `conda install -n base m2w64-toolchain`
- `$Env.PATH = C:\Users\user\miniconda3\Library\mingw-w64\bin`

This also has the advantage of being able to use GCC globally from the Windows command line after installation.
