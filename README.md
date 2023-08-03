# AgPdPtRu_ORR

This repository is tied to the manuscript *"A Flexible Theory for Catalysis: Learning Alkaline Oxygen Reduction on Complex Solid Solutions within the Ag-Pd-Pt-Ru Composition Space"* found at [DOI:10.1002/anie.202307187](https://doi.org/10.1002/anie.202307187) and serves to make the experimental and computational data publicly available.

To reproduce the data analysis and figures, it will be necessary to install the *utils* package. By running *00_get_dists_experiment.py* and *00_get_dists_grid.py* it is possible to obtain the adsorption energy distributions of \*OH and \*O, however the folder dist_libraries is also avaible for download at [ERDA](https://sid.erda.dk/cgi-sid/ls.py?share_id=Eh84Rp6A39) (4.11GB).

An overview of the repository:

- The *features* and *regression* folders hold the scripts *construct_feats.py* and *train_GNN.py*. These will create graph-features and train the GNN-model for adsorption energy inference.
- *00_get_dists_experiment.py* and *00_get_dists_grid.py* will simulate high-entropy alloy surrogate surfaces for both the experimentally samples compositions and for compositions uniformly distributed on a grid.
- *00_plot_volcano.py* recreates figure 3b.
- *01_plot_lsvs.py* recreates all plots of LSVs and adjusts the current densities of the AgPtRu materials library
- *02_shift_agptru.py* fits the discrepancy between materials libraries as described in the supplementary information of the manuscript.
- *03_plot_libraries.py* recreates both ternary and 3d plots.
- *04_fit_comp.py* fits the activities to the as-deposited alloy compositions.
- *05_fit_volcano.py* fits the activities using the theory-derived expression.
- *06_fit_histogram.py* fits the activities using SKlearn regression models.
- *07_plot_predictions.py* plots activity predictions in ternary plots.
- *08_flexible_comp.py* recreates the ternary plots included in figure 4.
