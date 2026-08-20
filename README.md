# µHSI → FTIRM Deep Learning Pipeline

Deep learning pipeline for translating hyperspectral microscopy (µHSI)
datacubes into Fourier transform infrared microspectroscopy (FTIRM)
biochemical maps or full FTIR spectra.

This repository accompanies the manuscript:

> Masjedi et al. **Predicting mid-infrared spectra and biochemical maps of unstained articular cartilage sections using visible-near-infrared hyperspectral microscopy**.

---

## Contents

- `uhsi_ftir_training.py` — Main training, evaluation, and Bayesian
  hyperparameter optimization pipeline.

---

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with ≥16 GB VRAM (tested on NVIDIA V100 and A100).
- **RAM**: ≥128 GB recommended; ≥200 GB required for full-scale spectra
  training without spatial downsampling.
- **Disk**: ≥50 GB free for a full 100-trial Bayesian optimization study
  (SQLite storage + per-trial artifacts).

### Software
Tested on Linux (CentOS 7 / Rocky 8) with:

- Python 3.9
- PyTorch 2.8.0 (CUDA 12.x)
- Optuna 4.8.0
- segmentation-models-pytorch 0.5.0
- scikit-learn 1.6.1
- scikit-image 0.24.0
- SciPy 1.13.1
- NumPy 2.0.2
- pandas 2.3.3
- h5py 3.14.0
- hdf5storage
- matplotlib
- tqdm

