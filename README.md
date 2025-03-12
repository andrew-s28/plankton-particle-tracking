# Scientific Analysis Code Template

This is the analysis code and datasets used for running Lagrangian particle tracking simulations in the Northern California Current System. 

# Usage

The primary Lagrangian particle tracking simulations are ran using [plankton_tracking.py](scripts/plankton_tracking.py). Run this script by:

1. `git clone https://github.com/andrew-s28/plankton-particle-tracking.git`

2. `uv run --script scripts/plankton_tracking.py`

      :warning: The [local install](scripts/parcels/) of [OceanParcels](https://github.com/OceanParcels/Parcels) is necessary to run this notebook since the fix implemented in [#1886](https://github.com/OceanParcels/Parcels/pull/1886) is not yet published on PyPi. If or when it is, this local install can be removed.
