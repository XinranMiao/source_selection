- `simulate_module.py`: functions for simulation.
- `simulate_splines.ipynb`: code to generate data based on splines.
- `bandit_simulation.ipynb`: code to run bandit selection on simulated data.
- `derived_data/`: output of `bandit_simulation.ipynb`, under the setting where we eliminate irrelevant sources (`conservative = False` in function `bandit_source_train()`).
- `conservative_data_path`: output of `bandit_simulation.ipynb`, under the setting similar to the manuscript (`conservative = True` in function `bandit_source_train()`).
- `vis.Rmd`: visualizing simulated data, with parameter `data_path` be one of `derived_data/` and `conservative_data_path`:

The following was written prior to March 2022.


This directory contains codes and output files of the simulation in section 4:

1. `simulation_functions.R`: basic functions of the data generation and experiments.
2. `run_bandit.R` and `run_ensemble.R`: implementation of bandit selection and ensemble methods. These files should be run inside `simulation_exeriment.R` where the variables are defined.
3. `simulation_experiment.R`: implementation of data generation and subset selection. It requires dependencies on `simulation_functions`, `run_bandit.R` and `run_ensemble.R`.
4. the directory of `./output/`: output files of the simulation experiment.
