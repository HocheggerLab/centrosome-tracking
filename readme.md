# Centrosome Tracking
This project includes the pipeline and some of the tools for exploring centrosome positioning data.
If you use this software for academic purposes, please cite:

*Stiff T, Echegaray-Iturra F et al. (2020), Prophase-specific perinuclear actin coordinates centrosome separation and positioning to ensure accurate chromosome segregation, Cell Reports.*

## Table of contents
* [Features](#features)
* [Status](#status)
* [Running from source](#Running from source)
* [Contact](#contact)
* [Licence](#licence)


## Features
List of features:
* Centrosome segmentation and tracking code in the imagej folder. It consists of a  Fiji(ImageJ) plugin for that uses Trackmate and some Python code to make an hdf5 file containing the raw image data and pandas dataframes per movie. The java project is configured independently into an Eclipse project.
* Dataset exploration software writen in Python and PyQt5.

## Status
This project stems from the work I've done in my Ph.D. studies, exclusively for the purpose of the associated publication. Consequently is in a state of _feature freeze_. Work is primarily focused on bug fixing and improving user experience.

## Running from source
Download the source archive, then execute in the Terminal:

    pip3 install -r requirements.txt
    
You can then run the application with:

    python3 run_gui_centrosome.py
    python3 run_gui_cell_boundary.py
    
### Dependencies
Requires Python 3.6 or greater and Java 1.8.0 or greater. For Java dependencies, see imagej/pom.xml. For Python dependencies, see requirements.txt.


## Contact
Created by [@fabioechegaray](https://twitter.com/fabioechegaray)
* [f.echegaray-iturra@sussex.ac.uk](mailto:f.echegaray-iturra@sussex.ac.uk)

## Licence
Centrosome Tracking

Copyright (C) 2020  The Hochegger Lab, Fabio Echegaray.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, in version 3 of the
License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.