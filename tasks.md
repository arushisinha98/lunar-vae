## Scientific Review: Moseley et al. (2020)

### Summary Assessment

This is a genuinely creative application of unsupervised learning to planetary thermophysics. The core contribution — using a VAE to disentangle thermophysical processes without a prior physics model — is well-motivated and the 4-orders-of-magnitude speedup over traditional inversion is compelling. However, several methodological concerns deserve attention before this approach matures into a primary analysis tool.

---

### Critical Methodological Concerns

**1. Training Data Bias and Generalization**

The VAE is trained on 48 hand-curated AOIs selected specifically because they exhibit interesting thermal behavior — craters of varying ages, pyroclastic deposits, and magnetic swirls. This introduces a survivorship bias: the model learns a latent space shaped by "interesting" regions rather than a representative sample of the lunar surface. The authors note that background profiles were deliberately downsampled to prevent domination of the training set, but this further skews the learned distribution. The consequence is visible in Figure 10, where the VAE fails to detect the cold spot around Chaplygin-B — a subtle but scientifically important anomaly type that was underrepresented in training.

A concrete improvement would be to stratify training data by surface type (mare, highland, fresh crater, mature crater, polar region) using existing geological maps, ensuring the latent space reflects the full surface distribution rather than a curated subset of anomalies.

**2. The β-VAE Hyperparameter is Under-Justified**

The authors use β = 0.2 without systematic justification. In β-VAEs, this parameter controls the trade-off between reconstruction fidelity and disentanglement quality — lower β prioritizes reconstruction, higher β prioritizes independence of latent dimensions. At β = 0.2, the model is biased toward reconstruction, which may explain why the latent variables are not fully disentangled (the authors acknowledge in Section 4 that solar onset, conductivity, and albedo bleed into each other). A systematic ablation study varying β across one or two orders of magnitude, evaluated on both reconstruction loss and correlation with physics inversion parameters, would substantially strengthen the paper's claims about disentanglement.

**3. The Physics Comparison is Asymmetric and Limited**

The physics inversion used for comparison is itself a simplified model: constant density, 1D heat equation, and a free effective albedo that absorbs topographic effects. Comparing the VAE's latent variables against parameters from this already-simplified model is a somewhat circular validation — you cannot use an imperfect model as a ground truth for evaluating another model. The authors should also compare against the full Hayne et al. (2017) H-parameter products more systematically across a broader spatial sample (they only show four craters in Figure 10), and report quantitative correlation statistics rather than relying on visual inspection.

**4. Temporal Information is Discarded**

Diviner collected data over nine years spanning different mission phases and orbital configurations. The preprocessing pipeline treats all measurements as a single time-averaged pool, discarding any temporal evolution signal. This means the model cannot distinguish between surface changes caused by, for example, fresh micrometeorite impacts that alter local thermal inertia over the observation period. Incorporating temporal structure — even as a simple year-binned additional input dimension — would add significant scientific value.

**5. Uncertainty Quantification is Abandoned**

The VAE's encoder produces both a mean and standard deviation for the approximate posterior distribution, but the authors explicitly discard the standard deviation at inference time. This is a substantial missed opportunity. The posterior variance is a direct, principled measure of the model's confidence at each surface location and could be used to flag profiles where the VAE is uncertain, separate from the L1 reconstruction loss. The loss map (Figure 9) conflates two distinct failure modes — profiles outside the training distribution versus profiles that are genuinely difficult to reconstruct — and the posterior variance could help disentangle these.

---

### Proposed Improvements (Actionable, Reasonable Timeframe)

**Improvement 1: Physically-Constrained Loss Function**

Encode known physical monotonicity constraints directly into the VAE's loss function. For instance, thermal inertia must be positive, albedo must be bounded between 0 and 1, and the onset delay must correlate with east-west slope aspect in a predictable direction. Adding soft penalty terms for violations of these constraints would guide the latent space toward physically interpretable solutions without requiring a full physics model. This is achievable in PyTorch with modest modifications to the training loop and would likely improve disentanglement more effectively than tuning β alone.

**Improvement 2: Disentanglement Metric-Driven Training**

Replace the current qualitative visual assessment of disentanglement with a quantitative metric such as the Mutual Information Gap (MIG) or DCI score, computed against the physics inversion parameters on the validation set. Use this metric as an early stopping criterion and for hyperparameter selection. This transforms the model evaluation from "does Figure 5 look physically plausible" to a measurable, reproducible benchmark.

**Improvement 3: Multi-Channel Input**

Incorporate channels 6, 7, and 8 simultaneously as a multi-channel 1D input rather than using only channel 7. The anisothermality signal across these channels is sensitive to rock abundance, a physically distinct surface property that the current model cannot access. The convolutional architecture requires only minor modification to accept multi-channel input, and this could enable the VAE to learn a fifth or sixth latent variable corresponding to rock fraction.

---

### Extension to Lacus Mortis

Lacus Mortis (45.0°N, 27.2°E) is a particularly well-chosen target for extending this methodology. It is a ~150 km flooded impact basin on the lunar nearside, containing the prominent Burg crater and a remarkable system of rilles (Lacus Mortis Rille), and sits at a latitude where Diviner has relatively good spatial coverage. Critically, it represents surface diversity — basaltic mare fill, highland rim material, an asymmetric fresh crater, and tectonic fractures — all within a compact spatial footprint. This makes it ideal for probing the limits of the VAE's disentanglement in a geologically complex region.

**Phase 1: Apply the Existing Methodology as a Baseline**

Extract temperature profiles from Lacus Mortis using the same preprocessing pipeline (channel 7, GP interpolation at 200×200 m bins, 0.5° bins). Run the pre-trained VAE directly on this region *without retraining* to generate latent maps. This is a zero-shot generalization test: if the model was truly learning generalizable thermophysical processes rather than features specific to its training AOIs, it should produce coherent latent maps over Lacus Mortis. Compare the latent 3 map (thermal inertia proxy) against the H-parameter map from Hayne et al. (2017) and against rock abundance estimates from Bandfield et al. (2011) for this region. Quantify agreement using the Pearson correlation and examine residuals spatially.

**Phase 2: Fine-Tuning and the Scientific Experiment**

The genuinely novel scientific experiment is to use the rille system as a natural laboratory for testing whether the VAE can detect subsurface structural heterogeneity. Rilles are extensional fractures that expose subsurface stratigraphy and can create localized variations in thermal inertia due to exposed bedrock, altered regolith compaction, and changes in surface roughness. The hypothesis is:

*Null hypothesis H₀: The VAE's latent 3 variable (thermal inertia proxy) shows no statistically significant difference between profiles extracted from within 2 km of the Lacus Mortis Rille and profiles from equivalent-slope, equivalent-latitude control points in the surrounding mare fill.*

*Alternative hypothesis H₁: Rille-proximal profiles show systematically elevated latent 3 values consistent with reduced regolith thickness or exposed higher-conductivity subsurface material.*

To test this, extract two matched profile populations: rille-proximal (within 2 km of the rille centerline, mapped from LROC imagery) and controls (same latitude band, same slope angle ±2°, >5 km from rille). Run both through the VAE and compare latent 3 distributions using a two-sample KS test and Mann-Whitney U test. Apply the physical interpretation transform (Î = e^(0.93·z3/2)) to convert to thermal inertia units for comparison with published values. This is a clean, falsifiable test that produces a scientifically meaningful result regardless of outcome.

**Phase 3: The Temporal Experiment**

Lacus Mortis is a relatively quiescent region with no confirmed recent impacts, making it suitable for a temporal analysis that the original paper did not attempt. Split the nine-year Diviner archive into three-year bins (2010–2013, 2013–2016, 2016–2019) and train three separate VAEs on the Lacus Mortis region alone. Compare the latent 3 maps across time periods. Any statistically significant drift in latent 3 values at specific locations — above the uncertainty estimated from the posterior standard deviation — would constitute evidence for genuine surface change and could be cross-referenced with the LROC image archive for confirmation of fresh impacts or mass wasting events near the Burg crater wall. This is a direct extension that the original paper explicitly identifies as future work ("detecting transient temperature anomalies from fresh impacts") and Lacus Mortis, with its steep Burg crater walls prone to mass wasting, is an ideal testbed.

---

### Overall Verdict

This is a solid proof-of-concept that establishes a viable research direction. The primary weaknesses are insufficient justification of key hyperparameters, a training distribution that may not generalize well, and underutilization of the model's own uncertainty outputs. The Lacus Mortis extension, particularly the rille-proximity experiment, would transform this from a methodology paper into a genuine discovery paper by posing a falsifiable geophysical hypothesis against which the VAE's latent representation can be directly tested.