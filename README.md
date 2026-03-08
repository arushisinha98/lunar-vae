# variational-autoencoder

This project ingests a carefully selected suite of nearly 2 million lunar surface temperature profiles, collected during the <a href = "https://www.jpl.nasa.gov/missions/diviner-lunar-radiometer-experiment-dlre">Diviner Lunar Radiometer Experiment</a>. The goal of this project is to train a Variational Autoencoder (VAE) on these profiles and to then explore the latent space created by the resultant model to understand if some physically informed relationships can and have been learned by the unsupervised model. A potential extention of this project involves introducing physically informed loss functions to further constrain and expedite this learning. This is currently a work in progress, incumbent upon the results of some physics-based/mechanistic models which will serve as the ground truth from which we may compute residuals.

The original model was trained on lunar surface temperature profiles of a select few areas of interest, chosen by Ben Moseley, the original author of this publication. Details on selection are outlined in Appendix B of <a href = "https://iopscience.iop.org/article/10.3847/PSJ/ab9a52"><i>Unsupervised Learning for Thermophysical Analysis on the Lunar Surface</i></a>.

In this repository, we recreate the methodology outlined in this publication with some refinements. We then set the stage for deploying the use of a trained VAE for the interpoation of lunar surface temperatures, specifically when observations at local noon (i.e. time of peak temperature) are missing. The accompanying slide deck can be used as a synopsis of this process. This VAE architecture was also trained on temperature profiles collected at and around <i>Lacus Mortis</i> but the results were not as promising, likely because the physical properties that we intended to learn demonstrated significantly lower variance in such a localized dataset.

---

## Usage

### 1. Retrain the original model

```bash
pixi install

# Train the original model
pixi run download_data.py
## Run Locally:
pixi run src/main.py # train the original model
## OR Submit a Job on HPC:
qsub train.pbs

## View Results
pixi run tensorboard --logdir results/summaries/<MODELNAME>
```

![Latent Space Profile & KL Divergence Loss](imgs/recreated_profiles_KLD.png)

*Figure 1: Top: Reconstructed lunar surface temperature profiles when varying one of the four latent dimensions at a time. Bottom: KL Divergence Loss on the original test (left) and train (right) datasets.*

![L1 L2 Loss](imgs/recreated_L1_L2.png)

*Figure 2: L1 (top) and L2 (bottom) loss curves showing the original model's convergence.*

### 2. Lacus Mortis extension

```bash
pixi install

## Run Locally:
pixi run src/lacus_mortis/preprocess.py
pixi run src/lacus_mortis/...

## OR Submit a Job on HPC:
qsub lacus.pbs
```