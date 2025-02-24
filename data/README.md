# Data README

Place your data subfolders in this folder.

## Risk-related files

For each calibration method, folders containing (1) the 1000 samples from the GAMs fit to the expected annual impact at all grid cells, given as NetCDF files, and (2) the optimal decision in each location identified using BDA for a given combination of inputs, given as csv files. The latter are not being used in the analyses (new files with optimal decision values are written elsewhere).
+ `ChangeFactor/`: change factor calibration
+ `UKCP_BC/`: bias corrected calibration
+ `UKCP_raw/`: no calibration

Exposure data:
+ `UKSSPs/`: files containing the Employment metric used to quantify the number of people working in outdoor physical jobs in the years 2020-2080

## Decision files

+ `decision_files_jit/`: Result of `default_dec_attr.py` -- optimal decision files resulting from varying all combinations of the risk parameters, resulting in 162 files
+ `decision_files_200/`: Result of `write_decision_files.py` -- optimal decision files resulting from varying all combinations of both risk and decision-related parameters, resulting in 32,400 files
