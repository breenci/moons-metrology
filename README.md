# Moons Metrology

This repository contains code and documentation used in the analysis and testing of the Multi-Object Optical and Near-Infrared Spectrograph (MOONS) focal plate metrology system. The system is based off a commercial off-the-shelf (COTS) solution, which has been customized for the specific needs of the MOONS project. A detailed description of the system architecture and components can be found at in the project sharepoint page at https://stfc365.sharepoint.com/:w:/r/sites/MOONSUKATC/Shared%20Documents/RFE/Metrology/Documentation/metrology_system_description/MOONS_metro_system_summary.docx?d=wc36ec59c2ea5438dae4c7345f3455021&csf=1&web=1&e=g5HaEJ

This repo contains helpful functions, scripts, and notebooks for working with metrology measurements of the MOONS focal plate. Helper functions can be found in the `src` directory. This directory contains the `transform`, `align`, `get_corr`, `metro_io` and `preprocess` modules.

Notebooks and scripts detailing specific workflows or examples can be found in the `testing_notebooks` or `testing_scripts` directories.

## Environment Setup

The simplest way to set up the correct conda environment for this project is using make. If you have make installed, you can run the following command in the terminal:

```console
make create_environment
```

Alternately the conda environment can be created directly from the environment.yml file in this repository. To do so, run the following command in the terminal:

```console
conda env create -f environment.yml
```

Both of these commands will create a conda environment called `moons-focus-plane` with the required dependencies installed. The environment can then be activated with the following command:

```console
conda activate moons-focus-plane
```

## Module Descriptions

The `transform` module contains functions for transforming between different coordinate systems used in the metrology system. It contains the `acq_cam`, `plate`, and `spatial` submodules. The `acq_cam` submodule deals with the acquisition camera coordinate system transformations, while the `plate` submodule handles transformations related to the focal plate coordinate system. The `spatial` submodule is responsible for more general coordinate transformations functions such as change of basis and projections.

The `align` module contains functions for aligning 3D point clouds using the Kabsch-Umeyama algorithm.

The `get_corr` module contains functions for finding correspondences between different point clouds.

The `metro_io` module contains functions for reading and writing metrology data in various formats.

The `pnt_filter` module contains functions for filtering subsets of point clouds based on various criteria, such as size or distance from focal plane.

