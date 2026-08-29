# Model Card — CyberPhotonics-SPR

## Model families

CyberPhotonics-SPR contains three research model families rather than one frozen pretrained model:

1. A conditioned PyTorch forward surrogate that maps PCF-SPR geometry plus analyte refractive index to sensing metrics.
2. A tandem inverse generator that maps target sensing metrics plus analyte refractive index to fabrication-constrained geometry.
3. Edge denoising and RI/resonance prediction models exported to full-INT8 TFLite/LiteRT when calibration data are supplied.

## Intended use

The models are intended for research prototyping, simulation-guided inverse design, active-learning candidate selection, and calibrated sensor-pipeline evaluation. Generated geometry must pass the repository fabrication constraints and should be independently verified with a trusted physics backend such as COMSOL before physical claims are made.

## Inputs and outputs

The conditioned forward model uses pitch, d/lambda, metal thickness, channel radius, and analyte RI. The inverse model consumes sensitivity, FOM, resonance wavelength, and analyte RI and returns the four geometry variables. Edge models consume resampled normalized spectra and return denoised spectra and RI/resonance estimates.

## Training and evaluation

Training code uses geometry-grouped splits to prevent points from the same fixed-geometry RI sweep leaking across train/validation partitions. Scientific validation includes target-satisfaction metrics, fabrication-violation rates, uncertainty/OOD diagnostics, and baseline comparisons. Edge validation evaluates the exported INT8 runtime rather than only the float model.

No performance number is hard-coded in this card. Reported accuracy, sensitivity, FOM, latency, confidence intervals, and hardware measurements must come from generated validation artifacts tied to a dataset/model hash.

## Limitations

Synthetic spectra validate software behavior but are not experimental evidence. COMSOL fidelity depends on the supplied `.mph` model and unit configuration. Inverse design is one-to-many, so geometry error is secondary to physical target satisfaction. OOD/confidence scores are decision aids, not probabilities of physical success. Hardware latency and sensor accuracy depend on the actual device and acquisition chain.

## Reproducibility

Use `scripts/create_reproducibility_bundle.py` to capture model hashes, dataset hashes, configuration, seed, Git state, and the complete installed Python-package snapshot for every reported experiment.
