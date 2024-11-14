# KaiB fold switching

This repository contains analysis scripts and notebooks for fold-switching of KaiB.

## Directory structure
- `dga` contains scripts pertaining to DGA simulations and analysis (kinetic and mechanistic quantities)
- `fixed_p_runs` contains scripts to run simulations with fixed proline isomerization state
- `iso_params` contains parameters for Upside to run proline isomerization with a double-well backbone potential. See `run_md3.py -p` for more information.
- `notebooks` contains all analysis notebooks
- `pdb` contains starting structures for simulations of various KaiB mutants
- `remd` contains scripts to run REMD simulations
- `scripts` contains basic analysis scripts using Upside's engine or various MD analysis packages (see [Requirements](#requirements) below)
- `src` contains Python files with useful loading and analysis functions. Mostly used in the `dga` analysis scripts and notebooks.

## Requirements
Upside2 ([https://github.com/sosnicklab/upside2-md/tree/master](https://github.com/sosnicklab/upside2-md/tree/master)) is
used for coarse-grained simulations. Follow the link for download and installation instructions. Upside requires
a working Python installation to run several of its scripts.

Python 3.11 was used for analysis. To recreate the conda environment,
see `requirements.yml` for the full list.
The important packages used are
- `numpy=1.25`
- `matplotlib=3.7`
- `seaborn`
- `scipy=1.11`
- `scikit-learn=1.3`
- `mdtraj=1.9.9`
- `pyemma=2.5`
- `pymbar>=4.0` for MBAR (to reweight REMD simulations)
- `ivac` for dimensionality reduction: [https://github.com/chatipat/ivac](https://github.com/chatipat/ivac) (see [here](https://pubs.acs.org/doi/full/10.1021/acs.jpcb.0c06477))
- `extq` for calculation of kinetic quantities: [https://github.com/chatipat/extq](https://github.com/chatipat/extq) (see [here](https://pmc.ncbi.nlm.nih.gov/articles/PMC8903024/) and [here](https://pubs.aip.org/aip/jcp/article-abstract/160/8/084108/3267205/Accurate-estimates-of-dynamical-statistics-using?redirectedFrom=fulltext))

Most packages (except `ivac` and `extq`) can be installed via `conda` or `pip`.
Plotting and some analysis was performed in Jupyter notebooks (`notebooks` directory).

## Scripts
General analysis scripts in `scripts` for computing 
energies, CVs, etc. Some of these rely on the Upside analysis scripts and so you will
need to include the appropriate path.

The scripts `run_all.py` and `run_md3.py` are loose wrapper scripts around the Upside
scripts and can be used to run either simple unbiased simulations or T-REMD simulations (see `remd`
directory for more details). These both operate with command line arguments; run 
`python run_all.py -h` for more details.

## Data
Data to reproduce plots and analysis has been deposited on [Zenodo](https://doi.org/10.5281/zenodo.14160033). The directory structure is similar to that of this repository and file names should correspond to those used in the Jupyter notebooks.

## Cite this work
If you found this work useful in your own research, please cite the following work:
```bibtex
@misc{zhang_temperature-dependent_2024,
	title        = {Temperature-{Dependent} {Fold}-{Switching} {Mechanism} of the {Circadian} {Clock} {Protein} {KaiB}},
	author       = {Zhang, Ning and Sood, Damini and Guo, Spencer C. and Chen, Nanhao and Antoszewski, Adam and Marianchuk, Tegan and Chavan, Archana and Dey, Supratim and Xiao, Yunxian and Hong, Lu and Peng, Xiangda and Baxa, Michael and Partch, Carrie and Wang, Lee-Ping and Sosnick, Tobin R. and Dinner, Aaron R. and LiWang, Andy},
	year         = 2024,
	publisher    = {bioRxiv},
	doi          = {10.1101/2024.05.21.594594},
	note         = {Pages: 2024.05.21.594594 Section: New Results},
}
```
