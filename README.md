# Plankton Particle Tracking in the Northern California Current System

This is the analysis code and datasets used for running Lagrangian particle tracking simulations in the Northern California Current System, specifically targeted at locations coincident with plankton sampling lines such as the [Newport Hydrographic Line](https://hmsc.oregonstate.edu/lab/newport-hydrographic-line-ecosystem-studies). 

## Usage

The primary Lagrangian particle tracking simulations are ran using [plankton_tracking.py](scripts/plankton_tracking.py). This script is designed to work with velocity fields from a [CROCO model](https://www.croco-ocean.org/). These model files are *far* too large to include in a repository. Set the path to model files [within the script](https://github.com/andrew-s28/plankton-particle-tracking/blob/main/scripts/plankton_tracking.py#L29) before running. 

Run this script by:

1. `git clone https://github.com/andrew-s28/plankton-particle-tracking.git`

2. `uv run scripts/plankton_tracking.py`

      :warning: The [local install](scripts/parcels/) of [OceanParcels](https://github.com/OceanParcels/Parcels) is necessary to run this notebook since the fix implemented in [#1886](https://github.com/OceanParcels/Parcels/pull/1886) is not yet published on PyPi. If or when it is, this local install can be removed.

      :warning: A C compiler is necessary to run this script. See below for install instructions.

## Installing a C Code Compiler

OceanParcels requires a C compiler (such as [GCC](https://gcc.gnu.org/)) in the path to run - i.e., `gcc --version` should successfully run from the command line. If using a Linux distro (or even WSL), this is straightforward: enjoy the ease with which `sudo apt install build-essential` handles things perfectly!

On Windows, however, there's a few more steps to installing a compiler. These are the steps I've taken to get it working on my machine:

1. Download latest [WinLibs](https://winlibs.com/) build from [this link](https://github.com/brechtsanders/winlibs_mingw/releases/download/15.1.0posix-12.0.0-msvcrt-r1/winlibs-x86_64-posix-seh-gcc-15.1.0-mingw-w64msvcrt-12.0.0-r1.zip)
2. Unzip and move to some reliable location (perhaps `%USERPROFILE%`)
3. Swap to the directory `%USERPROFILE%/winlibs-x86_64-posix-seh-gcc-15.1.0-mingw-w64msvcrt-12.0.0-r1/mingw64/bin`
4. Copy the file `gcc.exe` and rename `cc.exe` (this is a bit of a hack so that any `cc` calls in, e.g., Python source builds point to the up to date `gcc` executable). 
5. Add the `%USERPROFILE%/winlibs-x86_64-posix-seh-gcc-15.1.0-mingw-w64msvcrt-12.0.0-r1/mingw64/bin` directory to the Windows path (note you'll have to swap out `%USERPROFILE%` when actually adding it to the path).
