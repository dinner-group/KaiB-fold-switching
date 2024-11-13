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
See [https://github.com/sosnicklab/upside2-md](https://github.com/sosnicklab/upside2-md) for how to install the Upside MD engine.
The following packages are used for analysis and visualization:
- `python=3.11`
- `numpy`
- `scipy`
- `mdtraj`
- `MDAnalysis`
- `pyemma`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `seaborn`
- `ivac` for dimensionality reduction: [https://github.com/chatipat/ivac](https://github.com/chatipat/ivac) (see [here](https://pubs.acs.org/doi/full/10.1021/acs.jpcb.0c06477))
- `extq` for calculation of kinetic quantities: [https://github.com/chatipat/extq](https://github.com/chatipat/extq) (see [here](https://pmc.ncbi.nlm.nih.gov/articles/PMC8903024/) and [here](https://pubs.aip.org/aip/jcp/article-abstract/160/8/084108/3267205/Accurate-estimates-of-dynamical-statistics-using?redirectedFrom=fulltext))
- `pymbar` for REMD reweighting

Most packages (except `ivac` and `extq`) can be installed via `conda` or `pip`.

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
