# Citation Documents / Supporting References

## Image Generation & Noise Models

The synthetic data generation relies on physically grounded noise models to accurately simulate scanning electron microscope (SEM) drift, astigmatism, and noise:

1. **Gaussian and Poisson Noise Models in SEM Images**
   - *Reference*: Timischl, F., et al. (2012). "Noise models for scanning electron microscope images." *Scanning*, 34(5), 336-343.
   - *Justification*: Used to justify the speckle and combined shot noise (Poisson-Gaussian) models simulating the electron beam interaction variations in `generate_dataset.py`.

2. **Astigmatism and Barrel Distortion**
   - *Reference*: Reimer, L. (1998). *Scanning Electron Microscopy: Physics of Image Formation and Microanalysis*. Springer.
   - *Justification*: Astigmatism ratio and barrel distortion (k-factor) simulate magnetic lens imperfections.

3. **Charging Effects (Streak Artifacts)**
   - *Reference*: Cazaux, J. (2004). "Some considerations on the charging of insulators in scanning electron microscopy." *Scanning*, 26(4), 181-203.
   - *Justification*: Models the random streak artifacts applied in the data generation pipeline mimicking charge accumulation on dielectric layers.

## Deep Learning Verification

1. **Siamese Networks for Image Matching**
   - *Reference*: Bromley, J., et al. (1993). "Signature verification using a 'Siamese' time delay neural network." *Advances in Neural Information Processing Systems*.
   - *Justification*: The fundamental architecture for our patch-level verifier which decouples visual correlation from simple pixel intensity.

2. **Zero-mean Normalized Cross-Correlation (ZNCC)**
   - *Reference*: Lewis, J. P. (1995). "Fast normalized cross-correlation." *Vision interface*, 10(1), 120-123.
   - *Justification*: Used as the multi-scale candidate generation phase before CNN verification.
