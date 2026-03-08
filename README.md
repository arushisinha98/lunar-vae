# variational-autoencoder

This project ingests a carefully selected suite of nearly 2 million lunar surface temperature profiles, collected during the <a href = "https://www.jpl.nasa.gov/missions/diviner-lunar-radiometer-experiment-dlre">Diviner Lunar Radiometer Experiment</a>. The goal of this project is to train a Variational Autoencoder (VAE) on these profiles and to then explore the latent space created by the resultant model to understand if some physically informed trends can and have been learned by the unsupervised model. A potential extention of this project involves introducing physically informed loss functions to further constrain and expedite this learning. This is currently a work in progress, incumbent upon the results of some physics-based/mechanistic models which will serve as the ground truth from which we may compute residuals.

Lunar surface temperature profiles are of a select few craters that were deemed areas of interest by Ben Moseley. Details on selection are outlined in Appendix B of the following publication entitled <a href = "https://iopscience.iop.org/article/10.3847/PSJ/ab9a52"><i>Unsupervised Learning for Thermophysical Analysis on the Lunar Surface</i></a>. In this repository, we recreate the methodology outlined in this publication with some refinements. We then set the stage for deploying the use of a trained VAE for the interpoation of lunar surface temperatures, specifically when observations at local noon (i.e. time of peak temperature) are missing. The accompanying slide deck can be used as a synopsis of this process. This VAE architecture was also trained on temperature profiles collected at and around <i> Lacus Mortis </i> but the results were not as promising, most likely due to the fact that the physical properties that we intended to learn demonstrated significantly lower variance in such a localized dataset.

---

# Usage

```bash
pixi install

# Train the original model
pixi run download_data.py
## Run Locally:
pixi run src/main.py # train the original model
## OR Submit a Job on HPC:
qsub train.pbs

# Lacus Mortis extension
pixi run src/lacus_mortis/preprocess.py
```