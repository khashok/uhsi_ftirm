############################################################
# Map uHSI datacube (width × height × uhsi_ch) to FTIRM datacube (width x height x ftir_ch) or FTIRM-extracted biochecmial map (width x height).
# Hamidreza Masjedi, hmasjedi@uef.fi
# Biomedical Spectroscopy Laboratory
# Department of Technical Physics
# University of Eastern Finland
#
# This script performs:
#     1. Loading of paired µHSI (input) and FTIRM (target) data.
#     2. Preprocessing (baseline correction, Savitzky–Golay smoothing/
#        derivatives, per-sectrum normalization, feature scaling,
#        optional dimensionality reduction).p
#     3. Patch extraction with configurable input/output spatial sizes.
#     4. Training of a U-Net-like generator with optional conditional
#        PatchGAN adversarial regularization.
#     5. Bayesian hyperparameter optimization via Optuna, using
#        validation-only objectives (single- or multi-objective).
#     6. Evaluation with pixel-wise metrics (R^2, Pearson r, MAE, RMSE)
#        and, in spectra mode, an R^2 decomposition into global,
#        per-pixel spectral, and per-channel spatial components.
#
# 
# 
# This repository accompanies the manuscript:
#   Predicting mid-infrared spectra and biochemical maps of unstained articular cartilage sections using visible-near-infrared hyperspectral microscopy, Masjedi H. et al.
#
# Contact:
# Hamidreza Masjedi, hmasjedi@uef.fi
# Biomedical Spectroscopy Laboratory
# Department of Technical Physics
# University of Eastern Finland
# Date:    24.11.2025
############################################################

import optuna
import copy
import sys
import time
import gc
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse
import pickle
import numpy as np
import scipy.io as sio
from scipy.signal import savgol_filter
from skimage.morphology import remove_small_objects
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd  
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, PowerTransformer, QuantileTransformer, FunctionTransformer
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import json
import hdf5storage
from scipy.ndimage import zoom as ndimage_zoom
from scipy.ndimage import uniform_filter
from skimage.transform import resize
from scipy.ndimage import distance_transform_edt
import hashlib
import traceback as _tb
from matplotlib.widgets import Slider as _MplSlider
import multiprocessing as _mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

# Use the 'spawn' start method for child processes. This avoids POSIX
# fork() deadlocks that occur when OpenBLAS/MKL thread pools have
# already been initialized in the parent process (as happens once
# PyTorch or scikit-learn perform any matrix operation). 'spawn'
# creates a fresh Python interpreter per worker, so no mutex state is
# inherited. The overhead of re-importing top-level modules per worker
# is negligible relative to training time.

_MP_CTX_SPAWN = _mp.get_context("spawn")

np.unicode_ = np.str_
try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False
try:
    import segmentation_models_pytorch as smp
    _SMP_AVAILABLE = True
except Exception:
    _SMP_AVAILABLE = False

os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# Base seed used by DataLoader workers. Set by main() before training.
_WORKER_BASE_SEED = 0

# ============================================================
# FTIR wavenumber axis (cm^-1)
#
# The FTIRM instrument produces 623 evenly spaced spectral channels
# spanning 800.31–1999.82 cm^-1 (fingerprint region). This axis is
# hardcoded here rather than read from the .mat files to guarantee
# reproducibility across samples and to eliminate a source of
# inconsistency in earlier dataset versions. Any FTIR cube loaded
# by this pipeline is expected to have exactly 623 spectral channels.
# ============================================================

FTIR_WLS_OVERRIDE = np.linspace(
    8.003117418289185e+02,   # first value of your WN
    1.999815123558044e+03,   # last value of your WN
    623,                     # number of points
    dtype=np.float32,
)

def _worker_seed_init(worker_id):
    """Top-level (picklable) DataLoader worker initializer."""
    import random as _r
    seed = (_WORKER_BASE_SEED + worker_id) % (2**31 - 1)
    np.random.seed(seed)
    _r.seed(seed)
    torch.manual_seed(seed)


# ============================================================
#  Command-line interface
# ============================================================
def build_parser():
    ap = argparse.ArgumentParser(
        description="Map uHSI spectra to scalar/patch ground truth derived from FTIR maps."
    )

    # ----------------- Data I/O ----------------- #
    ap.add_argument(
        "--target_mode", type=str, default="map",
        choices=["map", "spectra"],
        help="Prediction target: 'map' = scalar FTIR-derived map; "
            "'spectra' = full FTIR spectrum per pixel/patch."
    )
    ap.add_argument(
        "--use_streaming", action=argparse.BooleanOptionalAction,
        default=True,
        help="Defer SG/spectrum_norm/x_scaler/PCA to DataLoader workers "
             "(per-patch). Saves RAM but slows GPU training because "
             "workers cannot keep up with the GPU. Only enable if the "
             "eager preprocessing path does not fit in memory.")

    ap.add_argument(
        "--data_dir", type=str, default=r"C:\Users\hmasjedi\OneDrive - University of Eastern Finland\Desktop\uhsi_ftir_data",
        help="Folder containing paired *_ftir_maps.mat and *_uhsi.mat files. "
             "Not required if --patch_dir is provided.")
    ap.add_argument(
        "--patch_dir", type=str, default=None,
        help="Directory with pre-extracted patch .npz files (fast path). "
             "If set, raw .mat loading and patch extraction are skipped.")
    ap.add_argument(
    "--results_dir", type=str, default=r"C:\Users\hmasjedi\OneDrive - University of Eastern Finland\Desktop\uhsi_ftir_data\results_amideI",
    help="Base directory where all results (per-fold subfolders, registry, "
         "LOSO summaries) will be saved. If not set, falls back to "
         "<patch_dir>/results or <data_dir>/results.")
    ap.add_argument(
        "--test_prefix", type=str, default="BDH03",
        help="Sample-name prefix used to select the TEST set "
             "(e.g. 'BDH07' matches BDH07_*).")
    ap.add_argument(
        "--val_prefix", type=str, default="BDH08",
        help="Sample-name prefix used to select the VALIDATION set "
             "(e.g. 'BDH08' matches BDH08_*). All other samples become TRAIN.")
    ap.add_argument(
        "--loso", action=argparse.BooleanOptionalAction, default=False,
        help="Run leave-one-sample-out cross-validation. Each sample is used as "
            "test once; --test_prefix and --val_prefix are ignored.")
    ap.add_argument(
        "--loso_val_strategy", type=str, default="next",
        choices=["next", "prev", "none"],
        help="How to choose the validation sample in each LOSO fold. "
            "'next' = next sample in list, 'prev' = previous, 'none' = no val set.")
    ap.add_argument(
        "--loso_only_samples", type=str, nargs="*", default=None,
        help="Optional: restrict LOSO to these test sample names (others stay in train).")


    # ----------------- Spatial downsampling ----------------- #
    ap.add_argument(
        "--spatial_scale", type=float, default=0.5,
        help="Spatial downsampling factor applied to uHSI/mask/FTIR maps. "
             "E.g. 0.3 reduces H and W to 30%%. Use 1.0 for no downsampling. "
             "Ignored when --patch_dir is set (must be done at patch extraction).")
    ap.add_argument(
        "--ds_method", type=str, default="area", choices=["bilinear", "area"],
        help="Downsampling method: 'area' (skimage resize with anti-aliasing, "
             "recommended) or 'bilinear' (scipy ndimage_zoom).")

    # ----------------- MAT-file field names ----------------- #
    ap.add_argument(
        "--uhsi_key", type=str, default="uhsi",
        help="Variable name of the uHSI cube inside the *_uhsi.mat file "
             "(stored as C x W x H, transposed to H x W x C at load).")
    ap.add_argument(
        "--exclude_wl_ranges", type=float, nargs="+", default=None,
        help="Wavelength ranges (nm) to EXCLUDE from the uHSI input cube, "
             "given as pairs: --exclude_wl_ranges LOW1 HIGH1 [LOW2 HIGH2 ...]. "
             "Example: --exclude_wl_ranges 700 730 removes 700-730 nm. "
             "Applied at load time, before any preprocessing. Ignored when "
             "--patch_dir is set (must be applied at patch extraction).")
    ap.add_argument(
        "--uhsi_wl_min", type=float, default=397.01,
        help="Minimum uHSI wavelength in nm (used to construct the axis).")
    ap.add_argument(
        "--uhsi_wl_max", type=float, default=1004.52,
        help="Maximum uHSI wavelength in nm (used to construct the axis).")
    ap.add_argument(
        "--amide_struct_key", type=str, default="R",
        help="Top-level struct name in the *_ftir_maps.mat file.")
    ap.add_argument(
        "--amide_field", type=str, default="AmideI",
        help="Field inside R used for cartilage masking fallback (e.g. 'AmideI').")
    ap.add_argument(
        "--map_field", type=str, default="AmideI",
        help="Field inside R used as the regression target (ground truth map). "
             "Available choices from dataset: 'AmideIItoAmideI', 'AmideII', 'AmideI', "
             "'Carbonate', 'Phosphate', 'Carbohydrate', 'XLR', 'Crystallinity', 'APS', "
             "'CarbohydratetoAmideI', 'MineralToMatrix', 'CarbonateToPhosphate', 'CarbonateToAmideI'.")    
    ap.add_argument(
        "--map_fields", type=str, nargs="+", default=["AmideI", "AmideII", "Carbohydrate", "AmideIItoAmideI", "CarbohydratetoAmideI"],
        help="Multiple FTIR map field names to predict simultaneously as "
             "channels of a multi-output UNet. Used when --model multi_unet. "
             "Example: --map_fields AmideI AmideII Carbonate MineralToMatrix. "
             "When set, overrides --map_field for target loading.")
    ap.add_argument(
        "--multi_unet_share_head", action=argparse.BooleanOptionalAction,
        default=False,
        help="If True, the multi-map UNet uses a single shared encoder/decoder "
             "with a multi-channel output head. If False, uses separate 1x1 "
             "conv heads per map (still shared backbone).")

    ap.add_argument(
        "--amide_map_field", type=str, default="map",
        help="Inner field name inside each FTIR map struct that holds the 2D array.")
    ap.add_argument("--ftir_key", type=str, default="data",
                    help="HDF5 key for FTIR cube in <name>_ftir.mat (target_mode=spectra).")
    ap.add_argument("--ftir_wls_key", type=str, default="WN",
                    help="Field with FTIR wavenumbers in *_ftir_maps.mat (target_mode=spectra).")

    # ----------------- Patch extraction ----------------- #
    ap.add_argument(
        "--avg_mode", type=str, default="patch",
        choices=["avg", "flat", "patch"],
        help="How to build model INPUT from each input patch: "
             "'avg' = average spectrum over masked pixels (shape: C), "
             "'flat' = flatten patch to 1D vector (shape: P*P*C), "
             "'patch' = keep 2D patch (shape: P, P, C). Use 'patch' for UNet/CNN2D.")
    ap.add_argument(
        "--patch", type=int, default=65,
        help="INPUT patch size (odd integer). Defines spatial context for the model.")
    ap.add_argument(
        "--overlap", type=float, default=0.2,
        help="Overlap fraction between adjacent training patches in [0, 1). "
             "Step = P * (1 - overlap). Higher = more samples but more redundancy.")
    ap.add_argument(
        "--min_in_mask_frac", type=float, default=0.1,
        help="Minimum fraction of cartilage (masked) pixels required in the INPUT patch "
             "for it to be kept. Use 1.0 to require fully-masked patches." )
    ap.add_argument(
        "--out_avg_mode", type=str, default="patch",
        choices=["avg", "flat", "patch"],
        help="How to build model OUTPUT target: "
             "'avg' = scalar (mean GT over mask), 'flat' = 1D vector (P*P), "
             "'patch' = 2D target map (P, P). Must match the model head.")
    ap.add_argument(
        "--out_patch", type=int, default=65,
        help="OUTPUT patch size (odd integer). Typically <= --patch.")
    ap.add_argument(
        "--min_out_mask_frac", type=float, default=0.1,
        help="Minimum fraction of cartilage (masked) pixels required in the OUTPUT patch.")
    ap.add_argument(
        "--der_mode", type=str, default="raw+der1+der2",
        choices=["raw", "der1", "der2", "raw+der1", "raw+der2",
                 "raw+der1+der2", "der1+der2"],
        help="Savitzky-Golay derivative features appended to the spectrum. "
             "Combinations increase channel count proportionally.")
    ap.add_argument("--out_der_mode", type=str, default="raw",
                choices=["raw", "der1", "der2", "raw+der1", "raw+der2",
                        "raw+der1+der2", "der1+der2"],
                help="Derivatives applied to spectral targets (target_mode=spectra).")

    # ----------------- Savitzky-Golay smoothing ----------------- #
    ap.add_argument(
        "--smooth_x", action=argparse.BooleanOptionalAction, default=False,
        help="If set, smooth input spectra with Savitzky-Golay BEFORE derivatives/DR.")
    ap.add_argument("--smooth_y", action=argparse.BooleanOptionalAction, default=False,
        help="Savitzky–Golay smoothing of spectral targets (target_mode=spectra).")
    ap.add_argument(
        "--sg_window", type=int, default=7,
        help="Savitzky-Golay window length (must be odd, auto-clipped to channel count).")
    ap.add_argument(
        "--sg_poly", type=int, default=2,
        help="Polynomial order for Savitzky-Golay smoothing (< sg_window).")

    # ----------------- Feature / target scaling ----------------- #
    ap.add_argument(
        "--x_scaler", type=str, default="standard",
        choices=["none", "standard", "minmax", "robust", "maxabs", "power", "quantile"],
        help="Scaler applied to INPUT features (spectra). "
             "Fitted on TRAIN, applied to VAL/TEST.")
    ap.add_argument(
        "--y_scaler", type=str, default="standard",
        choices=["none", "standard", "minmax", "robust", "maxabs", "power", "quantile", "log", "log_standard"],
        help="Scaler applied to TARGETS (y). 'log' uses log1p (non-negatives). "
             "'log_standard' chains log1p + StandardScaler.")
    ap.add_argument(
    "--spectrum_norm", type=str, default="none",
    choices=["none", "snv", "l2", "mean", "patch_snv", "snv+raw", "l2+raw"],
    help="Per-spectrum normalization applied independently to each "
        "spectrum/patch. Safe wrt test leakage. "
        "'snv+raw' and 'l2+raw' concatenate the normalized view with the "
        "raw spectrum along the channel axis (doubles C). This lets the "
        "model use both scatter-corrected and absolute-intensity views.")


    # ----------------- Dimensionality reduction ----------------- #
    ap.add_argument(
        "--dr_mode", type=str, default="none",
        choices=["none", "pca", "plsr_dr"],
        help="Spectral dimensionality reduction: "
             "'pca' (unsupervised), 'plsr_dr' (supervised, uses y_train), 'none'.")
    ap.add_argument(
        "--dr_components", type=int, default=32,
        help="Number of components kept after PCA / PLSR-DR.")

    # ----------------- Baseline correction ----------------- #
    ap.add_argument(
        "--baseline_mode", type=str, default="none",
        choices=["none", "offset", "emsc"],
        help="Spectral baseline correction: "
             "'offset' = subtract per-spectrum median, "
             "'emsc' = Extended Multiplicative Signal Correction.")
    ap.add_argument(
        "--emsc_degree", type=int, default=2,
        help="Polynomial degree for EMSC baseline (used only if baseline_mode='emsc').")


    # ----------------- Learning-rate scheduler ----------------- #
    ap.add_argument(
        "--scheduler", type=str, default="plateau",
        choices=["none", "cosine", "plateau", "onecycle"],
        help="LR scheduler strategy. 'plateau' reduces LR on val plateau, "
             "'cosine' = CosineAnnealingWarmRestarts, 'onecycle' = OneCycleLR.")
    ap.add_argument("--cosine_Tmax", type=int, default=50,
                    help="Cosine scheduler T_max (total or base period length).")
    ap.add_argument("--cosine_min_lr", type=float, default=1e-6,
                    help="Cosine scheduler minimum learning rate (eta_min).")
    ap.add_argument("--plateau_factor", type=float, default=0.5,
                    help="ReduceLROnPlateau: multiplicative LR decrease factor.")
    ap.add_argument("--plateau_patience", type=int, default=5,
                    help="ReduceLROnPlateau: epochs without improvement before LR cut.")
    ap.add_argument("--plateau_min_lr", type=float, default=1e-7,
                    help="ReduceLROnPlateau: lower bound for the learning rate.")
    ap.add_argument("--plateau_threshold", type=float, default=1e-4,
                    help="ReduceLROnPlateau: threshold for measuring new optimum.")

    # ----------------- Training hyperparameters ----------------- #
    ap.add_argument("--epochs", type=int, default=100,
                    help="Maximum training epochs (early stopping may trigger sooner).")
    ap.add_argument("--batch", type=int, default=8,
                    help="Mini-batch size for training and evaluation.")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="Initial (base) learning rate for AdamW.")
    ap.add_argument("--dropout", type=float, default=0.3,
                    help="Dropout probability used inside MLP/CNN/UNet blocks.")
    ap.add_argument(
        "--smooth_weight", type=float, default=0,
        help="Weight of spatial smoothness penalty in the loss (0 = disabled). "
             "Only meaningful for patch outputs.")
    ap.add_argument(
        "--patience_early_stop", type=int, default=15,
        help="Patience for early stopping (epochs without improvement). "
            "Set to 0 or a negative value to DISABLE early stopping.")
    ap.add_argument(
        "--max_shift", type=int, default=3,
        help="Shift-tolerant loss: tolerated target misregistration in pixels "
             "(searches best shift in [-k, +k]^2). 0 disables.")

    # ----------------- Loss function ----------------- #
    ap.add_argument(
        "--loss", type=str, default="huber",
        choices=["mse", "huber", "l1", "wmse"],
        help="Masked regression loss. 'huber'/SmoothL1 is robust to outliers, "
             "'l1' predicts the conditional median, 'wmse' weights rare y-bins higher.")
    ap.add_argument("--huber_delta", type=float, default=1.0,
                    help="Beta/delta of SmoothL1 (Huber) loss.")
    ap.add_argument("--wmse_nbins", type=int, default=20,
                    help="Histogram bins used by 'wmse' to estimate y frequency.")
    ap.add_argument("--wmse_power", type=float, default=0.5,
                    help="Exponent on inverse frequency weights: "
                         "0 = uniform, 1 = 1/freq (stronger re-weighting for larger values).")
    
    # ----------------- Sharpness-promoting auxiliary losses ----------------- #
    ap.add_argument(
        "--grad_weight", type=float, default=0.0,
        help="Weight of the SPATIAL GRADIENT (Sobel) matching loss. "
            "Penalizes |∂x pred - ∂x gt| + |∂y pred - ∂y gt| over the cartilage mask. "
            "This is the most effective single term against blurry predictions, "
            "because it directly forces predicted edges/transitions to match GT. "
            "0 disables. Typical useful range: 0.1-1.0. Only applied to patch outputs.")
    ap.add_argument(
        "--lap_weight", type=float, default=0.0,
        help="Weight of the LAPLACIAN (second-derivative) matching loss. "
            "Penalizes |∇² pred - ∇² gt|, i.e. mismatch in local curvature / "
            "high-frequency content. Complements --grad_weight by enforcing "
            "fine-detail similarity beyond first-order edges. "
            "0 disables. Typical useful range: 0.05-0.5. Patch outputs only.")
    ap.add_argument(
        "--var_weight", type=float, default=0,
        help="Weight of the LOCAL STANDARD-DEVIATION matching loss. "
            "Computes per-pixel local std (in a --var_kernel window) for both "
            "prediction and GT, and penalizes |std_pred - std_gt|. This punishes "
            "predictions that are locally flat (the classic symptom of a "
            "model 'hedging' by outputting a smooth mean). "
            "0 disables. Typical useful range: 0.1-0.5. Patch outputs only.")
    ap.add_argument(
        "--var_kernel", type=int, default=5,
        help="Window size (odd integer) used by the local-variance loss "
            "(--var_weight) to compute local mean and standard deviation. "
            "Smaller windows enforce sharper texture matching; larger windows "
            "match more global texture. Should be << --out_patch. Default: 5.")
    ap.add_argument(
        "--fft_weight", type=float, default=0.0,
        help="Weight of the FREQUENCY-DOMAIN (FFT magnitude) matching loss. "
            "Computes |FFT(pred)| - |FFT(gt)| (mean absolute difference) so the "
            "predicted patch shares the same spectral energy distribution as the "
            "GT patch — including high frequencies that pixel-wise losses "
            "average out. Use sparingly; can interact with --max_shift and small "
            "patches. 0 disables. Typical useful range: 0.01-0.1.")
    ap.add_argument(
        "--spatial_loss_summary", type=str, default="mean",
        choices=["mean", "bands"],
        help="How to collapse 4D predictions/targets (B,P,P,C_out) into a 2D map "
            "for max_shift search and edge-enhanced (gradient/Laplacian/var/FFT) "
            "losses. 'mean' averages all spectral channels; 'bands' averages over "
            "indices given by --spatial_loss_bands.")
    ap.add_argument(
        "--spatial_loss_bands", type=int, nargs="*", default=None,
        help="Spectral channel indices used to build the 2D summary when "
            "--spatial_loss_summary=bands. Example: 120 240 360 (Amide I/II "
            "neighbourhoods). Indices are clamped to [0, C_out-1].")
    ap.add_argument(
        "--per_channel_spatial", action=argparse.BooleanOptionalAction,
        default=False,
        help="If set (spectra mode), apply spatial edge losses "
            "(--grad_weight, --lap_weight, --var_weight, --fft_weight) "
            "to EACH spectral channel independently and average, instead "
            "of applying them to a 2D spectral summary. Has no effect in "
            "map mode.")
    ap.add_argument(
        "--spectral_smooth_weight", type=float, default=0.0,
        help="Weight of the SPECTRAL-AXIS smoothness penalty "
            "(mean squared first-order differences along the wavenumber "
            "axis of the prediction). Only meaningful in target_mode="
            "spectra. 0 disables. Typical useful range: 1e-4 to 1e-2.")

    # ----------------- Model selection ----------------- #
    ap.add_argument(
        "--model", type=str, default="unet",
        choices=["mlp", "plsr", "cnn", "cnn1d", "unet", "unet_simple",
                 "pretrained_unet", "multi_unet"],
        help="Regression model. 'unet'/'cnn' require --avg_mode patch. "
             "'mlp','cnn1d','plsr' typically use --avg_mode avg (per-pixel spectrum).")
    
    # ----------------- Pretrained model hyperparameters ----------------- #
    ap.add_argument("--pretrained_encoder", type=str, default="efficientnet-b3",
        help="smp encoder name (resnet34, efficientnet-b0..b3, etc.). "
            "Used when --model pretrained_unet.")
    ap.add_argument("--pretrained_weights", type=str, default="imagenet",
        help="smp encoder weights ('imagenet' or 'none').")
    ap.add_argument("--freeze_encoder_epochs", type=int, default=5,
        help="Train only decoder+head for this many warmup epochs, "
            "then unfreeze encoder for fine-tuning. 0 = no freezing.")
    ap.add_argument("--finetune_lr_mult", type=float, default=0.1,
        help="LR multiplier for the pretrained encoder once unfrozen "
            "(typical fine-tuning value: 0.1).")
    
    # ----------------- UNet hyperparameters ----------------- #
    ap.add_argument("--unet_base_ch", type=int, default=64,
                    help="Base channel count of the first UNet encoder stage.")
    ap.add_argument("--unet_depth", type=int, default=4,
                    help="Number of downsampling stages in the UNet (2^depth stride).")

    # ----------------- MLP hyperparameters ----------------- #
    ap.add_argument(
        "--mlp_hidden", type=int, nargs="+", default=[64, 256, 32, 8],
        help="Sequence of hidden-layer widths for the MLP (space-separated).")

    # ----------------- CNN2D hyperparameters ----------------- #
    ap.add_argument(
        "--cnn_hidden", type=int, nargs="+", default=[512, 2048, 512, 128, 32],
        help="Per-layer output channel counts for the 2D CNN backbone.")

    # ----------------- CNN1D hyperparameters ----------------- #
    ap.add_argument("--cnn1d_hidden", type=int, default=64,
                    help="Number of channels in each 1D-CNN spectral conv layer.")
    ap.add_argument("--cnn1d_layers", type=int, default=3,
                    help="Number of stacked 1D conv layers.")

    # ----------------- PLSR hyperparameters ----------------- #
    ap.add_argument("--plsr_components", type=int, default=10,
                    help="Number of latent components for PLSR regression.")

    # ----------------- Extra spatial/context channels ----------------- #
    ap.add_argument(
        "--add_depth_channel", action=argparse.BooleanOptionalAction, default=False,
        help="Append normalized cartilage depth (per-pixel, in [0,1]) as an extra "
            "input channel. Computed from the cartilage mask via distance transform.")
    ap.add_argument(
        "--add_pos_channels", action=argparse.BooleanOptionalAction, default=False,
        help="Append normalized (row, col) coordinates of each pixel within the "
            "full image as 2 extra input channels in [0,1].")
    ap.add_argument(
        "--add_global_channel", action=argparse.BooleanOptionalAction, default=False,
        help="Append a low-resolution version of the full uHSI image (mean over "
            "wavelength) as an extra input channel, cropped per patch.")
    ap.add_argument(
        "--global_lowres_scale", type=float, default=0.1,
        help="Downsampling factor used to build the low-res global image "
            "(only used if --add_global_channel).")
    ap.add_argument(
        "--global_channel_pcs", type=int, default=0,
        help="If >0, use this many PCA components of the global low-res cube as "
            "extra channels (instead of a single mean-over-wavelength channel).")
    ap.add_argument(
        "--global_mode", type=str, default="full_image_resize",
        choices=["neighborhood", "full_image_resize"],
        help="How the low-res global channel(s) are built. "
            "'neighborhood' = downsample uHSI by --global_lowres_scale, "
            "upsample back to the (downsampled) image grid, then crop a "
            "patch-sized window per patch (sees local blurred context). "
            "'full_image_resize' = resize the entire uHSI image directly to "
            "(patch x patch) and use that same global view for every patch "
            "of the sample (sample-level conditioning). Combine with "
            "--add_pos_channels to keep 'where am I' information.")
    ap.add_argument(
        "--nonmask_fill", type=str, default="constant",
        choices=["keep_zero", "zero", "constant", "patch_mean"],
        help="How to fill non-cartilage pixels in the INPUT uHSI cube "
            "BEFORE baseline correction / scaling. 'constant' (default, "
            "recommended) = fill with a fixed value; combined with "
            "--add_mask_channel this yields a leak-free fold-aware "
            "representation. 'zero' = same as constant with value 0. "
            "'keep_zero' = original behaviour. 'patch_mean' = replace "
            "with the sample's cartilage-mean spectrum; this injects a "
            "SAMPLE-CONDITIONED constant into fold pixels and can leak "
            "sample identity through the receptive field. Provided ONLY "
            "for ablations; not allowed by the BO search space.")
    ap.add_argument(
        "--nonmask_fill_value", type=float, default=0.0,
        help="Constant value used when --nonmask_fill=constant. Default 0 means "
            "the model sees zeros at fold pixels in the raw cube; after the "
            "input scaler is fit on cartilage pixels, these become approximately "
            "(0 - mean) / std, i.e. a neutral background.")
    ap.add_argument(
        "--add_mask_channel", action=argparse.BooleanOptionalAction, default=True,
        help="Append the cartilage mask as an extra binary input channel (0/1).")

    # ----------------- Reproducibility ----------------- #
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for numpy and torch (affects init, shuffling).")


    # ----------------- GAN / CycleGAN options ----------------- #
    ap.add_argument(
        "--gan_mode", type=str, default="pix2pix",
        choices=["none", "pix2pix", "pix2pix_inv"],
        help="Enable adversarial training. "
             "'pix2pix' = conditional PatchGAN with shift-tolerant L1 "
             "(Isola et al., 2017). "
             "'pix2pix_inv' = pix2pix plus an inverse-consistency "
             "reconstruction term ||G_B(G_A(x)) - x||_1 on cartilage "
             "pixels. Inspired by CycleGAN [Zhu et al., 2017], but "
             "applied to PAIRED data: it acts as a representation-"
             "preserving regulariser, not as an unsupervised domain-"
             "translation constraint. Only applied when in_patch == "
             "out_patch.")
    ap.add_argument(
        "--gan_lambda_l1", type=float, default=100.0,
        help="Weight of the supervised (shift-tolerant) L1/regression loss "
            "in pix2pix / pix2pix_inv. Standard pix2pix uses 100.")
    ap.add_argument(
        "--gan_lambda_adv", type=float, default=1.0,
        help="Weight of the adversarial generator loss. Standard = 1.")
    ap.add_argument(
        "--gan_lambda_cycle", type=float, default=10.0,
        help="Weight of the inverse-consistency reconstruction term "
             "||G_B(G_A(x)) - x||_1 in pix2pix_inv mode. Despite the "
             "name (kept for backwards compatibility), this is NOT "
             "CycleGAN cycle consistency, because the data is paired. "
             "Standard value: 10.")
    ap.add_argument(
        "--gan_lambda_identity", type=float, default=0.0,
        help="DEPRECATED. Identity loss G(real_B) ≈ real_B is not "
             "well-defined when input and output domains have different "
             "channel counts (uHSI vs FTIR). Kept only for backwards "
             "compatibility; any non-zero value is ignored.")
    ap.add_argument(
        "--gan_d_base_ch", type=int, default=64,
        help="Base channel count for the PatchGAN discriminator.")
    ap.add_argument(
        "--gan_d_n_layers", type=int, default=3,
        help="Number of conv layers in the PatchGAN discriminator (each "
             "downsamples by 2). Typical: 3 (70x70 receptive field).")
    ap.add_argument(
        "--gan_d_lr", type=float, default=2e-4,
        help="Discriminator learning rate. Typical: 2e-4 with beta1=0.5.")
    ap.add_argument(
        "--gan_d_beta1", type=float, default=0.5,
        help="Adam beta1 for both G and D in GAN training. Standard = 0.5.")
    ap.add_argument(
        "--gan_loss_type", type=str, default="lsgan",
        choices=["lsgan", "vanilla", "hinge"],
        help="Adversarial loss type. 'lsgan' (recommended, stable), "
             "'vanilla' (BCE), 'hinge'.")
    ap.add_argument(
        "--gan_d_update_every", type=int, default=2,
        help="Update discriminator every N generator steps. Raise to 2-5 if "
             "D dominates G (G loss diverges, D loss collapses to 0).") 
    ap.add_argument(
        "--gan_shift_aware", action=argparse.BooleanOptionalAction, default=True,
        help="If True, the supervised L1 inside pix2pix / pix2pix_inv uses "
             "--max_shift to tolerate misregistration between uHSI and "
             "FTIR. This is the principled way to handle unknown sub-"
             "pixel registration error in paired data.")


    # ----------------- Cartilage mask (fallback) ----------------- #
    ap.add_argument(
        "--amide_thresh", type=float, default=0.0,
        help="AmideI threshold used to build cartilage mask ONLY if no "
             "saved 'cartilage_mask' is present in the FTIR file.")
    ap.add_argument(
        "--min_obj_size", type=int, default=200,
        help="Minimum connected-component size (pixels) kept when building "
             "the fallback cartilage mask via AmideI threshold.")


    # ----------------- Bayesian optimization ----------------- #
    ap.add_argument("--bo", action=argparse.BooleanOptionalAction, default=True,
                    help="Run Bayesian hyperparameter optimization with Optuna.")
    ap.add_argument("--bo_trials", type=int, default=100,
                    help="Number of Optuna trials.")
    ap.add_argument("--bo_proxy", type=str, default="single_fold",
                    choices=["single_fold", "subset_loso", "full_loso"],
                    help="Proxy used to evaluate each trial during BO.")
    ap.add_argument("--bo_subset_size", type=int, default=3,
                    help="Folds used per trial when bo_proxy=subset_loso.")
    ap.add_argument("--bo_epochs", type=int, default=50,
                    help="Reduced epochs used during BO trials.")
    ap.add_argument(
        "--bo_metric", type=str, default="pareto_r2_blur",
        choices=["auto", "val_r2", "val_loss",
                "val_r2_minus_blur", "val_r2_spectral",
                "pareto_r2_blur"],
        help="Validation-only metric to optimize. 'auto' picks "
            "'val_r2_minus_blur' for target_mode=map and "
            "'val_r2_spectral' for target_mode=spectra. "
            "'pareto_r2_blur' runs MULTI-OBJECTIVE NSGA-II optimization "
            "with (maximize val_r2, minimize val_blur_pen); no scalar "
            "blur weight is needed. Test-set metrics are NEVER allowed "
            "as BO objectives.",
    )
    ap.add_argument("--bo_blur_penalty", type=float, default=0.3,
                    help="Weight of the blur penalty in 'val_r2_minus_blur' "
                        "(score = val_R2 - w * mean|1 - local_std_ratio|). "
                        "Both terms are evaluated on the VALIDATION sample in "
                        "physical units to keep the BO objective scaler-independent.")
    ap.add_argument("--bo_storage", type=str, default=None,
                    help="Optional Optuna storage URL (e.g. sqlite:///bo.db) for "
                        "resuming. Default: in-memory study (lost on restart).")
    ap.add_argument(
        "--bo_trial_timeout_sec", type=int, default=9000,
        help="Maximum wall-clock seconds for a single BO trial, measured "
             "from the start of preprocessing through model evaluation. "
             "Trials exceeding this budget are pruned (state=PRUNED) so "
             "the study can move on. 0 disables the budget. Default: "
             "9000 (2.5 h).")



    return ap

def parse_args():
    return build_parser().parse_args()


def _suggest_hparams(trial, args):
    """Mutate args in place with Optuna-suggested hyperparameters."""
    # The hyperparameter search space defined below reflects the final
    # configuration used in the manuscript. Ablations of alternative search
    # spaces (e.g. GAN mode as a categorical variable, or loss family as a
    # categorical variable) are documented in Supplementary Table S2 and
    # were performed in separate studies with modified versions of this
    # function.

    args.max_shift = trial.suggest_int("max_shift", 0, 2)
    args.lr            = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    args.dropout       = trial.suggest_float("dropout", 0.0, 0.5)

    _gan_active = args.gan_mode != "none"
    # if _gan_active:
    #     args.gan_lambda_l1   = trial.suggest_float(
    #         "gan_lambda_l1", 50.0, 150.0, log=True)
    #     args.gan_lambda_adv  = trial.suggest_float(
    #         "gan_lambda_adv", 0.2, 2.0, log=True)
    #     if args.gan_mode == "pix2pix_inv":
    #         args.gan_lambda_cycle = trial.suggest_float(
    #             "gan_lambda_cycle", 2.0, 20.0, log=True)
    #     args.gan_d_lr        = trial.suggest_float(
    #         "gan_d_lr", 5e-5, 5e-4, log=True)
    #     args.gan_loss_type   = trial.suggest_categorical(
    #         "gan_loss_type", ["lsgan"])
    #     args.gan_d_update_every = trial.suggest_int(
    #         "gan_d_update_every", 1, 4)
    #     args.gan_shift_aware = True



    # Preserve multi_unet across BO trials when the user selected it on the CLI.
    # If the user set --model multi_unet, BO must keep that. Otherwise default
    # to the original single-output unet search space.
    args.model = "unet"
    if args.model == "pretrained_unet":
        args.pretrained_encoder = trial.suggest_categorical("pretrained_encoder", ["efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "resnet34"],)
        args.freeze_encoder_epochs = trial.suggest_int("freeze_encoder_epochs", 0, 8)
        args.finetune_lr_mult      = trial.suggest_float("finetune_lr_mult", 0.05, 0.5, log=True)
    elif args.model in ("unet", "unet_simple", "multi_unet"):
        args.unet_base_ch = trial.suggest_categorical("unet_base_ch", [16, 32, 64, 128])
        args.unet_depth   = trial.suggest_int("unet_depth", 2, 4)
    # Patch sizes can only be tuned when extracting from raw data.
    if getattr(args, "patch_dir", None) is None:
        if args.model == "pretrained_unet":
            # smp encoders need H,W >= 32 to avoid pathological reflect-pad
            # and to allow the 5 stride-2 stages of ResNet/EfficientNet.
            args.patch = trial.suggest_int("patch", 33, 129, step=2)
        else:
            args.patch = trial.suggest_int("patch", 17, 129, step=2)
        if args.patch % 2 == 0:
            args.patch += 1
        args.out_patch = trial.suggest_int("out_patch", 17, args.patch, step=2)
        if args.out_patch % 2 == 0:
            args.out_patch += 1
        # AFTER args.patch and args.out_patch are finalized, cap the
    # PatchGAN depth so the 4x4 convs actually fit.
    if _gan_active:
        op = int(args.out_patch)

        # ---- Max safe PatchGAN depth as a function of out_patch ----
        # PatchGAN does n stride-2 4x4 convs then 2 stride-1 4x4 convs.
        # Each stride-2 conv halves the size; each stride-1 conv reduces by 1.
        # We need the spatial size to remain >= 4 after the stride-1 stack.
        if op >= 65:
            d_hi = 4          # ~70x70 receptive field (classic pix2pix)
        elif op >= 33:
            d_hi = 3
        elif op >= 17:
            d_hi = 2
        else:
            d_hi = 1
        d_lo = min(2, d_hi)

        args.gan_d_n_layers = trial.suggest_int(
            "gan_d_n_layers", d_lo, d_hi
        )

        # ---- NEW: tune discriminator base channels ----
        # Standard PatchGAN uses 64; smaller (32) reduces D capacity and
        # often stabilises training when the generator is small.
        args.gan_d_base_ch = trial.suggest_categorical(
            "gan_d_base_ch", [32, 64, 96]
        )


    args.batch         = 8
    # args.loss          = trial.suggest_categorical("loss", ["mse", "huber", "l1"])
    args.loss = "huber"
    # if args.loss == "wmse":
    #     args.wmse_nbins = trial.suggest_int("wmse_nbins", 10, 50, step=5)
    #     args.wmse_power = trial.suggest_float("wmse_power", 0.0, 1.5)
    if args.loss == "huber":
        args.huber_delta = trial.suggest_float("huber_delta", 0.1, 2.0)
    args.overlap       = trial.suggest_float("overlap", 0.0, 0.3)
    # Sharpness-promoting auxiliary loss. We keep ONLY the spatial gradient
    # (Sobel) term in the BO search space. Laplacian, local-variance and
    # FFT-magnitude losses are mathematically correlated with the gradient
    # term (they all penalize discrepancies in spatial derivatives /
    # high-frequency content) and including all four simultaneously creates
    # a degenerate, multi-objective search space that BO cannot resolve
    # cleanly. Restricting BO to grad_weight gives a single, interpretable
    # sharpness regularizer; the others remain available on the CLI for
    # ablation studies but are fixed to 0 during BO.
    args.grad_weight = trial.suggest_float("grad_weight", 0.0, 1.0)
    args.lap_weight  = 0.0
    args.var_weight  = 0.0
    args.fft_weight  = 0.0
    if args.target_mode == "spectra":
        args.per_channel_spatial = trial.suggest_categorical(
            "per_channel_spatial", [False, True])
        args.spectral_smooth_weight = trial.suggest_float(
            "spectral_smooth_weight", 1e-5, 1e-2, log=True)
    else:
        args.per_channel_spatial = False
        args.spectral_smooth_weight = 0.0

    if args.target_mode == "spectra":
        # In spectra mode, restrict DR to {none, pca}. PLSR-DR requires
        # a scalar supervised target per spectrum, which is not
        # naturally defined for full-spectrum FTIR outputs.
        args.dr_mode = trial.suggest_categorical("dr_mode", ["none", "pca"])
    else:
        args.dr_mode = trial.suggest_categorical(
            "dr_mode", ["none", "pca"])
    if args.dr_mode in ("pca", "plsr_dr"):
        args.dr_components = trial.suggest_int(
            "dr_components", 8, 64, step=8)
    args.scheduler      = trial.suggest_categorical("scheduler", ["none", "cosine", "plateau"])
    # if args.scheduler == "plateau":
    #     args.plateau_factor = trial.suggest_float("plateau_factor", 0.1, 0.7)
    #     args.plateau_patience = trial.suggest_int("plateau_patience", 3, 15)
    if args.scheduler == "cosine":
        # Cap T_max at the BO trial budget so cosine restarts are
        # actually realized during the trial; otherwise the schedule
        # degenerates to a near-linear warmup and gives no signal.
        _t_max_hi = max(20, int(getattr(args, "bo_epochs", 50)))
        args.cosine_Tmax = trial.suggest_int("cosine_Tmax", 20, _t_max_hi)
        # args.cosine_min_lr = trial.suggest_float(
        #     "cosine_min_lr", 1e-8, 1e-5, log=True)
    args.der_mode      = trial.suggest_categorical("der_mode", ["raw", "raw+der1", "raw+der2", "der1+der2", "raw+der1+der2"])

    args.smooth_x       = trial.suggest_categorical("smooth_x", [False, True])
    # if args.target_mode == "spectra":
    #     args.smooth_y   = trial.suggest_categorical("smooth_y", [False, True])
    _sg_active = (
        args.smooth_x
        or args.der_mode != "raw"
        or (args.target_mode == "spectra" and getattr(args, "smooth_y", False))
    )
    if _sg_active:
        args.sg_window = trial.suggest_categorical("sg_window", [5, 7, 9, 11, 13])
        # args.sg_poly = trial.suggest_int("sg_poly", 2, 4)

    # Sample spectrum_norm BEFORE x_scaler so that the conditional x_scaler
    # search space below uses the value chosen for THIS trial, not a stale
    # carry-over from the previous trial / CLI default.
    args.spectrum_norm = trial.suggest_categorical(
        "spectrum_norm", ["none", "snv", "snv+raw"])

    # Use distinct parameter names per branch so each name has a
    # FIXED value space (Optuna requirement for persistent storage).
    if args.spectrum_norm == "snv":
        args.x_scaler = trial.suggest_categorical(
            "x_scaler_snv", ["none", "power", "quantile"]
        )
    else:
        args.x_scaler = trial.suggest_categorical(
            "x_scaler_nonsnv",
            ["none", "standard", "minmax", "robust",
            "maxabs", "power", "quantile"],
        )





    if args.target_mode == "spectra":
        # Drop log/log_standard: FTIR absorbance can be slightly negative
        # (baseline residuals, derivatives), and the clip-to-zero in our
        # log path is a non-invertible step that changes a non-trivial
        # fraction of pixels. Keep the linear/monotone families only.
        args.y_scaler = trial.suggest_categorical(
            "y_scaler",
            ["standard", "robust", "maxabs", "power", "quantile"],
        )
    else:
        args.y_scaler = trial.suggest_categorical(
            "y_scaler",
            ["standard", "robust", "maxabs", "power", "quantile",
            "log", "log_standard"],
        )
    args.baseline_mode = trial.suggest_categorical("baseline_mode", ["none", "offset", "emsc"])
    if args.baseline_mode == "emsc":
        args.emsc_degree = trial.suggest_int("emsc_degree", 1, 4)

    # Aux channels & fold handling — only meaningful for patch-input
    # UNet/CNN models with 4D inputs.
    if args.avg_mode == "patch":
        # Spatial aux channels are meaningful only with 2D patch inputs.
        # args.add_depth_channel  = trial.suggest_categorical(
        #     "add_depth_channel",  [False, True])
        # args.add_pos_channels   = trial.suggest_categorical(
        #     "add_pos_channels",   [False, True])
        # args.add_global_channel = trial.suggest_categorical(
        #     "add_global_channel", [False, True])
        # if args.add_global_channel:
        #     args.global_mode = trial.suggest_categorical(
        #         "global_mode", ["neighborhood", "full_image_resize"])
        args.add_depth_channel  = True
        args.add_pos_channels   = True
        args.add_global_channel = False
    else:
        # Force-off so non-patch trials cannot accidentally request
        # aux channels that are undefined for their input layout.
        args.add_depth_channel  = False
        args.add_pos_channels   = False
        args.add_global_channel = False

    # Methodology safety: BO must not select sample-conditioned fold fill.
    if getattr(args, "nonmask_fill", "constant") == "patch_mean":
        args.nonmask_fill = "constant"

    return args


@lru_cache(maxsize=4)
def _load_raw_cube_cached(path, key):
    """Per-process cached loader for raw uHSI cubes. Reused across BO
    trials within the same worker. Bounded at 4 entries to prevent OOM
    on full_loso."""
    with h5py.File(path, "r") as f:
        return f[key][()].astype(np.float32).transpose(2, 1, 0)

def r2_components_spectra(y_true, y_pred, mask, eps=1e-12):
    """
    y_true, y_pred : (N_pixels, C_ftir) after applying the cartilage mask.
    mask           : (N_pixels,) bool, used only to verify shape upstream.

    Returns dict with:
        r2_global   : flat R^2 over all (pixel, wavenumber) entries.
        r2_spectral : mean over pixels of per-pixel R^2 (variance computed
                      across wavenumbers within each pixel).
        r2_spatial  : mean over wavenumbers of per-wavenumber R^2 (variance
                      computed across pixels within each channel).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # --- Global (flattened) ---
    yt_flat = y_true.ravel()
    yp_flat = y_pred.ravel()
    ss_res = np.sum((yt_flat - yp_flat) ** 2)
    ss_tot = np.sum((yt_flat - yt_flat.mean()) ** 2)
    r2_global = 1.0 - ss_res / (ss_tot + eps) if ss_tot > 0 else np.nan

    # --- Per-pixel (spectral shape fidelity) ---
    # For each pixel, compute R^2 across wavenumbers.
    mu_pix = y_true.mean(axis=1, keepdims=True)            # (N, 1)
    ss_tot_pix = np.sum((y_true - mu_pix) ** 2, axis=1)    # (N,)
    ss_res_pix = np.sum((y_true - y_pred) ** 2, axis=1)    # (N,)
    valid = ss_tot_pix > eps
    r2_per_pixel = np.full(y_true.shape[0], np.nan)
    r2_per_pixel[valid] = 1.0 - ss_res_pix[valid] / ss_tot_pix[valid]
    r2_spectral = float(np.nanmean(r2_per_pixel)) if valid.any() else np.nan

    # --- Per-wavenumber (spatial fidelity per channel) ---
    mu_ch = y_true.mean(axis=0, keepdims=True)             # (1, C)
    ss_tot_ch = np.sum((y_true - mu_ch) ** 2, axis=0)      # (C,)
    ss_res_ch = np.sum((y_true - y_pred) ** 2, axis=0)     # (C,)
    valid_ch = ss_tot_ch > eps
    r2_per_ch = np.full(y_true.shape[1], np.nan)
    r2_per_ch[valid_ch] = 1.0 - ss_res_ch[valid_ch] / ss_tot_ch[valid_ch]
    r2_spatial = float(np.nanmean(r2_per_ch)) if valid_ch.any() else np.nan

    return {
        "r2_global":   r2_global,
        "r2_spectral": r2_spectral,
        "r2_spatial":  r2_spatial,
    }

def _score_from_metrics(metrics, key, blur_penalty=0.3, _from_bo=False):
    """
    Compute the score that a BO objective should consume.

    `_from_bo=True` is set by run_bo's objective(). It enforces the
    protocol that no test-set quantity can be used as a BO objective;
    test metrics are retained here only for offline logging.
    """

    # 1. Strict Validation Metrics (Used by Optuna)
    if key == "val_r2":
        return metrics.get("val_r2_best", -1e9)
    if key == "val_loss":
        return -metrics.get("val_loss_best", 1e9)
    if key == "val_r2_minus_blur":
        return metrics.get("val_r2_minus_blur_best", -1e9)
    if key == "val_r2_spectral":
        return metrics.get("val_r2_spectral_best", -1e9)

    # 2. Test metrics — only allowed for offline logging.
    if _from_bo and key in ("test_r2", "test_r2_minus_blur"):
        raise ValueError(
            f"bo_metric='{key}' uses test-set quantities, which are not "
            f"permitted as BO objectives under the leak-free protocol. "
            f"Use one of {{'val_r2', 'val_loss', 'val_r2_minus_blur'}}."
        )


    if key == "test_r2":
        d = metrics.get("test_r2_unscaled_per_sample", {})
        vals = [v for v in d.values() if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else -1e9

    if key == "test_r2_minus_blur":
        d_r2 = metrics.get("test_r2_unscaled_per_sample", {})
        r2_vals = [v for v in d_r2.values()
                   if v is not None and np.isfinite(v)]
        r2_mean = float(np.mean(r2_vals)) if r2_vals else -1e9

        d_sr = metrics.get("test_local_std_ratio_per_sample", {})
        sr_vals = [v for v in d_sr.values()
                   if v is not None and np.isfinite(v)]
        blur_pen = float(np.mean([abs(1.0 - v) for v in sr_vals])) \
            if sr_vals else 0.0
        return r2_mean - blur_penalty * blur_pen

    return -1e9

def _pareto_objectives_from_metrics(metrics):
    """
    Return (val_r2, val_blur_pen) for multi-objective BO.

    val_r2      : best per-epoch validation R^2 in PHYSICAL units
                  (maximize).
    val_blur_pen: best per-epoch blur penalty mean|1 - local_std_ratio|
                  in PHYSICAL units (minimize). 0 means perfect
                  local-texture preservation.

    Both quantities are validation-only and scaler-independent, so the
    same leak-free contract as the scalarized objectives applies.
    """
    val_r2 = metrics.get("val_r2_best", None)
    if val_r2 is None or not np.isfinite(val_r2):
        val_r2 = -1e9

    blur = metrics.get("val_blur_pen_best", None)
    if blur is None or not np.isfinite(blur):
        blur = 1e9

    return float(val_r2), float(blur)


def run_bo(args, samples, results_base):
    # Choose folds for proxy
    if args.bo_proxy == "single_fold":
        # Use the prefix-based split (test_prefix / val_prefix) as a single fold
        tr, va, te = _build_splits_prefix(samples, args.test_prefix, args.val_prefix)
        bo_folds = [("bo_single", tr, va, te)]
    elif args.bo_proxy == "subset_loso":
        # Use first K samples as test rotations
        K = min(args.bo_subset_size, len(samples))
        bo_folds = []
        for i in range(K):
            tr, va, te = _build_splits_loso(samples, i, args.loso_val_strategy)
            bo_folds.append((f"bo_loso_{samples[i]['name']}", tr, va, te))
    else:  # full_loso
        bo_folds = []
        for i in range(len(samples)):
            tr, va, te = _build_splits_loso(samples, i, args.loso_val_strategy)
            bo_folds.append((f"bo_loso_{samples[i]['name']}", tr, va, te))

    bo_results_base = os.path.join(results_base, "bo")
    os.makedirs(bo_results_base, exist_ok=True)

    # Decide objective shape ONCE, outside objective(), so it is
    # captured by closure rather than recomputed per fold.
    _is_pareto = (args.bo_metric == "pareto_r2_blur")

    def objective(trial):
        if _is_pareto and trial.number == 0:
            print("[BO] Multi-objective Pareto mode active "
                "(maximize val_r2, minimize val_blur_pen).")

        trial_args = copy.deepcopy(args)
        trial_args = _suggest_hparams(trial, trial_args)
        trial_args.epochs = args.bo_epochs
        # Mark this as a BO trial so run_one_fold skips heavy/redundant saves.
        trial_args._is_bo_trial = True

        # In single-objective mode `scores` holds per-fold scalars.
        # In Pareto mode `obj_pairs` holds per-fold (r2, blur) tuples.
        scores = []
        obj_pairs = []

        for fold_tag, tr, va, te in bo_folds:
            tag = f"trial{trial.number:04d}_{fold_tag}"
            s = None
            try:
                metrics = run_one_fold(
                    args=trial_args, samples=samples,
                    train_samples=tr, val_samples=va, test_samples=te,
                    fold_tag=tag, results_base=bo_results_base,
                )

                if _is_pareto:
                    r2, blur = _pareto_objectives_from_metrics(metrics)
                    if (not np.isfinite(r2) or not np.isfinite(blur)
                            or r2 < -10.0):
                        raise optuna.TrialPruned()
                    obj_pairs.append((r2, blur))
                else:
                    s = _score_from_metrics(
                        metrics,
                        args.bo_metric,
                        blur_penalty=getattr(args, "bo_blur_penalty", 0.3),
                        _from_bo=True,
                    )
                    if s is None or not np.isfinite(s) or s < -10.0:
                        raise optuna.TrialPruned()

            except optuna.TrialPruned:
                raise
            except torch.cuda.OutOfMemoryError as e:
                print(f"[BO] CUDA OOM in trial {trial.number}, "
                      f"fold {fold_tag}: {e}")
                _tb.print_exc()
                try: gc.collect()
                except Exception: pass
                if torch.cuda.is_available():
                    try: torch.cuda.empty_cache()
                    except Exception: pass
                    try: torch.cuda.ipc_collect()
                    except Exception: pass
                raise optuna.TrialPruned()
            except RuntimeError as e:
                msg = str(e).lower()
                hard_cuda = any(k in msg for k in (
                    "illegal memory access", "device-side assert",
                    "cublas", "cudnn", "an illegal",
                ))
                _tb.print_exc()
                if hard_cuda and torch.cuda.is_available():
                    try: torch.cuda.synchronize()
                    except Exception: pass
                    try: torch.cuda.empty_cache()
                    except Exception: pass
                raise optuna.TrialPruned()
            except Exception:
                _tb.print_exc()
                raise optuna.TrialPruned()
            finally:
                _ctx_poisoned = False
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except torch.cuda.OutOfMemoryError as _e:
                        print(f"[BO cleanup] CUDA OOM during sync "
                              f"(recoverable): {_e}")
                    except RuntimeError as _e:
                        msg = str(_e).lower()
                        if "out of memory" in msg:
                            print(f"[BO cleanup] CUDA OOM during sync "
                                  f"(recoverable): {_e}")
                        else:
                            print(f"[BO cleanup] CUDA sync raised hard "
                                  f"error: {_e}")
                            _ctx_poisoned = True
                try: gc.collect()
                except Exception: pass
                if torch.cuda.is_available():
                    try: torch.cuda.empty_cache()
                    except Exception: pass
                    try: torch.cuda.ipc_collect()
                    except Exception: pass
                    try: torch.cuda.reset_peak_memory_stats()
                    except Exception: pass
                if _ctx_poisoned:
                    print("[BO] CUDA context poisoned — stopping study; "
                          "relaunch the script to resume from sqlite storage.")
                    raise optuna.exceptions.OptunaError(
                        "CUDA context poisoned; relaunch to resume."
                    )

            # ---- Per-fold aggregation (single-objective only) ----
            if not _is_pareto:
                scores.append(s)
                trial.report(np.mean(scores), step=len(scores))
                if trial.should_prune():
                    raise optuna.TrialPruned()

        # ---- Return objective(s) for this trial ----
        if _is_pareto:
            if not obj_pairs:
                raise optuna.TrialPruned()
            r2_arr   = np.array([p[0] for p in obj_pairs], dtype=np.float64)
            blur_arr = np.array([p[1] for p in obj_pairs], dtype=np.float64)
            return float(r2_arr.mean()), float(blur_arr.mean())

        if not scores:
            raise optuna.TrialPruned()
        return float(np.mean(scores))

    # ---- Build study name + create study ----
    if (getattr(args, "model", "") == "multi_unet"
            and getattr(args, "map_fields", None)):
        study_suffix = "multi-" + "-".join(args.map_fields)
    else:
        study_suffix = args.map_field

    _bo_id = (
        f"target-{args.target_mode}"
        f"_gan-{args.gan_mode}"
        f"_loss-{args.loss}"
        f"_metric-{args.bo_metric}"
    )

    if _is_pareto:
        # Multi-objective: maximize val_r2, minimize val_blur_pen.
        # NSGA-II is population-based; pruning is not defined here.
        study = optuna.create_study(
            directions=["maximize", "minimize"],
            sampler=optuna.samplers.NSGAIISampler(seed=args.seed),
            storage=getattr(args, "_bo_storage_obj", args.bo_storage),
            study_name=f"uhsi_map_bo_pareto_{study_suffix}_{_bo_id}",
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
            storage=getattr(args, "_bo_storage_obj", args.bo_storage),
            study_name=f"uhsi_map_bo_sharp_{study_suffix}_{_bo_id}",
            load_if_exists=True,
        )

    study.optimize(
        objective,
        n_trials=None,
        timeout=71 * 3600,
        gc_after_trial=True,
        callbacks=[
            optuna.study.MaxTrialsCallback(
                args.bo_trials,
                states=(optuna.trial.TrialState.COMPLETE,),
            )
        ],
    )

    # ---- Post-optimisation: pick best (or knee) and persist ----
    if _is_pareto:
        pareto_trials = [
            t for t in study.best_trials
            if t.values is not None
            and all(np.isfinite(v) for v in t.values)
        ]
        print(f"\n=== Pareto front: {len(pareto_trials)} non-dominated "
              f"trial(s) ===")

        if pareto_trials:
            r2_vals   = np.array([t.values[0] for t in pareto_trials])
            blur_vals = np.array([t.values[1] for t in pareto_trials])

            def _norm(a, maximize):
                lo, hi = float(a.min()), float(a.max())
                if hi - lo < 1e-12:
                    return np.zeros_like(a, dtype=np.float64)
                n = (a - lo) / (hi - lo)
                return (1.0 - n) if maximize else n

            r2_d   = _norm(r2_vals,   maximize=True)
            blur_d = _norm(blur_vals, maximize=False)
            dist   = np.sqrt(r2_d ** 2 + blur_d ** 2)
            knee_idx = int(np.argmin(dist))
            knee_trial = pareto_trials[knee_idx]

            print(f"  Knee-point trial #{knee_trial.number}:")
            print(f"    val_r2       = {knee_trial.values[0]:.4f}")
            print(f"    val_blur_pen = {knee_trial.values[1]:.4f}")
            print(f"    params       = {knee_trial.params}")

            front_path = os.path.join(bo_results_base, "pareto_front.json")
            with open(front_path, "w") as f:
                json.dump({
                    "knee_trial_number": knee_trial.number,
                    "knee_values": list(knee_trial.values),
                    "knee_params": knee_trial.params,
                    "front": [
                        {"trial_number": t.number,
                         "values": list(t.values),
                         "params": t.params}
                        for t in pareto_trials
                    ],
                }, f, indent=2)
            print(f"Saved Pareto front to {front_path}")

            try:
                all_trials = [
                    t for t in study.trials
                    if t.values is not None
                    and all(np.isfinite(v) for v in t.values)
                ]
                all_r2   = [t.values[0] for t in all_trials]
                all_blur = [t.values[1] for t in all_trials]

                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(all_r2, all_blur, s=18, c="0.6",
                           alpha=0.6, label="All trials")
                ax.scatter(r2_vals, blur_vals, s=40, c="C0",
                           edgecolor="k", linewidth=0.5,
                           label="Pareto front")
                ax.scatter([knee_trial.values[0]],
                           [knee_trial.values[1]],
                           s=120, marker="*", c="C3",
                           edgecolor="k", linewidth=0.6,
                           label="Knee point (selected)")
                ax.set_xlabel(r"Validation $R^2$ (maximize)")
                ax.set_ylabel(r"Validation blur penalty (minimize)")
                ax.set_title("Bayesian optimisation: fidelity-sharpness "
                             "Pareto frontier")
                ax.legend(loc="best", fontsize=9)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig_path = os.path.join(bo_results_base, "pareto_front.png")
                fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved Pareto scatter to {fig_path}")
            except Exception as e:
                print(f"[BO] Pareto plot failed: {e}")
        else:
            knee_trial = None
            print("[BO] Pareto front is empty; no trial selected.")

        study._uhsi_knee_trial = knee_trial
    else:
        try:
            print("\n=== Best trial ===")
            print(f"  value : {study.best_value:.4f}")
            print(f"  params: {study.best_params}")

            best_path = os.path.join(bo_results_base, "best_params.json")
            with open(best_path, "w") as f:
                json.dump({"value": study.best_value,
                           "params": study.best_params}, f, indent=2)
            print(f"Saved best params to {best_path}")
        except ValueError:
            print("[BO] No completed trial; nothing to save.")

    return study

# ----------------- Per-spectrum / per-patch normalization ----------------- #

def apply_spectrum_norm(X, mode="none", eps=1e-6):
    """
    Per-spectrum normalization computed independently per spectrum
    (no train/test leakage).

    Supported modes:
        "none"      : no change
        "snv"       : Standard Normal Variate (per-spectrum z-score)
        "l2"        : per-spectrum L2 normalization
        "mean"      : subtract per-spectrum mean only
        "patch_snv" : SNV computed per patch (requires 4D)
        "snv+raw"   : concatenate [SNV(x), x] along the channel axis,
                      doubling the channel count. Lets the model use
                      both scatter-corrected and absolute-intensity
                      views simultaneously.
        "l2+raw"    : concatenate [L2(x), x] along the channel axis.

    Works with:
        X.ndim == 2 : (N, C)
        X.ndim == 4 : (N, P, P, C)
    """
    if mode == "none":
        return X

    X = np.asarray(X, dtype=np.float32)

    # Stacked modes: produce one normalized view alongside the raw view.
    if mode in ("snv+raw", "l2+raw"):
        base = mode.split("+", 1)[0]
        X_norm = apply_spectrum_norm(X, mode=base, eps=eps)
        return np.concatenate([X_norm, X], axis=-1).astype(np.float32)

    if X.ndim == 2:
        if mode == "snv":
            mu = X.mean(axis=1, keepdims=True)
            sd = X.std(axis=1, keepdims=True)
            zero_std_count = int((sd <= eps).sum())
            sd = np.where(sd > eps, sd + eps, 1.0)
            apply_spectrum_norm._last_zero_std_count = zero_std_count
            if zero_std_count > 0 and os.environ.get("UHSI_SNV_VERBOSE", "0") == "1":
                print(f"[SNV] {zero_std_count} spectra had std<={eps}; "
                    f"replaced with 1.0 (mean-centered only).")
            return ((X - mu) / sd).astype(np.float32)
        elif mode == "l2":
            n = np.linalg.norm(X, axis=1, keepdims=True)
            n = np.where(n > eps, n + eps, 1.0)
            return (X / n).astype(np.float32)
        elif mode == "mean":
            return (X - X.mean(axis=1, keepdims=True)).astype(np.float32)
        elif mode == "patch_snv":
            raise ValueError("patch_snv requires 4D input (N,P,P,C)")
        else:
            raise ValueError(f"Unknown spectrum_norm mode: {mode}")

    elif X.ndim == 3:
        # (P, P, C) single patch — used by streaming dataset.
        # Fast path: compute per-spectrum mu/sd directly on the 3D view.
        P1, P2, C = X.shape
        if mode == "snv":
            mu = X.mean(axis=-1, keepdims=True)
            sd = X.std(axis=-1, keepdims=True)
            sd = np.where(sd > eps, sd + eps, 1.0)
            return ((X - mu) / sd).astype(np.float32)
        elif mode == "l2":
            n = np.linalg.norm(X, axis=-1, keepdims=True)
            n = np.where(n > eps, n + eps, 1.0)
            return (X / n).astype(np.float32)
        elif mode == "mean":
            return (X - X.mean(axis=-1, keepdims=True)).astype(np.float32)
        elif mode == "patch_snv":
            X_re = X.reshape(1, -1)
            mu = X_re.mean(axis=1, keepdims=True)
            sd = X_re.std(axis=1, keepdims=True)
            sd = np.where(sd > eps, sd + eps, 1.0)
            return ((X_re - mu) / sd).reshape(P1, P2, C).astype(np.float32)
        else:
            raise ValueError(f"Unknown spectrum_norm mode: {mode}")


    elif X.ndim == 4:
        N, P1, P2, C = X.shape
        if mode in ("snv", "l2", "mean"):
            X_flat = X.reshape(-1, C)
            X_norm = apply_spectrum_norm(X_flat, mode=mode, eps=eps)
            return X_norm.reshape(N, P1, P2, C)
        elif mode == "patch_snv":
            X_re = X.reshape(N, -1)
            mu = X_re.mean(axis=1, keepdims=True)
            sd = X_re.std(axis=1, keepdims=True)
            sd = np.where(sd > eps, sd + eps, 1.0)
            X_re = (X_re - mu) / sd
            return X_re.reshape(N, P1, P2, C).astype(np.float32)
        else:
            raise ValueError(f"Unknown spectrum_norm mode: {mode}")

    else:
        raise ValueError(f"Unsupported X.ndim={X.ndim}")


# ----------------- Console log + metrics helpers ----------------- #


def build_aux_maps(uhsi_cube, mask, args, depth_mask=None):
    """
    Build per-sample auxiliary 2D maps that will later be cropped per patch
    and appended as extra input channels.

    Parameters
    ----------
    uhsi_cube : (H, W, C)
    mask : (H, W) bool — used for pos/global channel normalization (final_mask)
    args : argparse Namespace
    depth_mask : (H, W) bool, optional — used ONLY for the depth channel
                 (cartilage_mask). Falls back to `mask` if None.
    """
    if depth_mask is None:
        depth_mask = mask

    H, W, C = uhsi_cube.shape
    full_maps = []
    patch_maps = []
    P = int(args.patch)

    # 0) Cartilage mask channel (binary 0/1; follows the patch crop)
    if getattr(args, "add_mask_channel", False):
        full_maps.append(mask.astype(np.float32))

    # 1) Normalized cartilage depth (per-pixel, uses depth_mask = cartilage_mask)
    if args.add_depth_channel:
        depth = cartilage_depth(depth_mask.astype(bool))
        depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
        full_maps.append(depth)


    # 2) Normalized (row, col) coordinates (per-pixel, follows the patch crop)
    if args.add_pos_channels:
        rr = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None].repeat(W, axis=1)
        cc = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :].repeat(H, axis=0)
        full_maps.append(rr)
        full_maps.append(cc)

    # 3) Low-resolution global image
    # 3) Low-resolution global image
    if args.add_global_channel:
        mode = getattr(args, "global_mode", "neighborhood")

        # Per-image PCA auxiliary channels would fit a per-sample basis
        # on val/test cartilage spectra, which is inconsistent with the
        # rest of the pipeline (all unsupervised preprocessing steps
        # are fit on training data only). This option is therefore
        # disabled here. A leak-free implementation would fit a single
        # PCA on TRAIN cartilage pixels and reuse it for val/test; this
        # is not implemented in the current version, so we fall back to
        # the mean-image channel.
        use_pca_aux = bool(args.global_channel_pcs > 0)
        if use_pca_aux:
            print("[build_aux_maps] NOTE: --global_channel_pcs > 0 is "
                  "disabled in this build because the basis would be fit "
                  "per image (including val/test). Falling back to a "
                  "single per-channel mean image. A leak-free variant "
                  "would fit a single PCA on TRAIN cartilage pixels "
                  "and reuse it across all splits.")
            use_pca_aux = False

        if mode == "full_image_resize":
            mean_img = uhsi_cube.mean(axis=-1)
            img = resize(mean_img, (P, P), order=1,
                         anti_aliasing=True, preserve_range=True)
            img = img.astype(np.float32)
            mask_pp = resize(mask.astype(np.float32), (P, P),
                             order=0, anti_aliasing=False,
                             preserve_range=True) > 0.5
            sel = mask_pp & np.isfinite(img)
            if sel.any():
                mu = float(img[sel].mean())
                sd = float(img[sel].std() + 1e-6)
                img = (img - mu) / sd
            # Zero out non-cartilage region so it does not pretend to be a strong
            # negative cartilage value at the fold border.
            img = np.where(mask_pp, img, 0.0).astype(np.float32)
            patch_maps.append(img)


        else:
            # mode == "neighborhood"
            s = float(args.global_lowres_scale)
            s = max(min(s, 1.0), 0.01)
            H_lo, W_lo = max(1, int(H * s)), max(1, int(W * s))

            min_eff = min(H_lo, W_lo)
            if min_eff < P / 4:
                print(f"[build_aux_maps] WARNING: global_mode=neighborhood "
                      f"with H_lo={H_lo}, W_lo={W_lo} but patch P={P}. "
                      f"The per-patch global crop will be nearly flat "
                      f"(effective resolution under patch ~ "
                      f"{min_eff * P / max(H, W):.1f} pixels). Consider "
                      f"--global_mode full_image_resize or raising "
                      f"--global_lowres_scale.")

            mean_img = uhsi_cube.mean(axis=-1)
            lo = resize(mean_img, (H_lo, W_lo), order=1,
                        anti_aliasing=True, preserve_range=True)
            hi = resize(lo, (H, W), order=1,
                        anti_aliasing=False, preserve_range=True)
            sel = mask.astype(bool) & np.isfinite(hi)
            if sel.any():
                mu = float(hi[sel].mean())
                sd = float(hi[sel].std() + 1e-6)
                hi = (hi - mu) / sd
            full_maps.append(hi.astype(np.float32))

    return {"full_maps": full_maps, "patch_maps": patch_maps}

class _Tee:
    """Write stdout/stderr to both terminal and a log file."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def _save_metrics_json(results_dir, metrics):
    """Write metrics.json, replacing NaN/Inf with None for valid JSON."""
    def clean(v):
        if isinstance(v, float):
            return v if np.isfinite(v) else None
        return v
    out = {k: clean(v) for k, v in metrics.items()}
    path = os.path.join(results_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Metrics saved to: {path}")


def cartilage_depth(mask):
    mask = mask.astype(bool)
    dist = distance_transform_edt(mask).astype(np.float32)
    dist[~mask] = np.nan
    denom = np.nanmax(dist)
    if not np.isfinite(denom) or denom <= 0:
        return np.full_like(dist, np.nan)
    return dist / denom


def depth_profile(values, depth_map, mask, n_bins=50):
    bins = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    mean_prof = np.full(n_bins, np.nan)
    std_prof  = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = (
            (depth_map >= bins[i]) &
            (depth_map < bins[i+1]) &
            mask &
            np.isfinite(values)
        )
        if np.any(sel):
            v = values[sel]
            mean_prof[i] = v.mean()
            std_prof[i]  = v.std()
    return centers, mean_prof, std_prof

def plot_depth_profiles(gt, pred, mask, sample_name, out_dir):
    depth = cartilage_depth(mask)

    z, gt_mean,   gt_std   = depth_profile(gt,   depth, mask)
    _, pred_mean, pred_std = depth_profile(pred, depth, mask)

    plt.figure(figsize=(4, 5))
    # GT
    plt.plot(gt_mean, z, color="black", linewidth=2, label="GT")
    plt.fill_betweenx(z, gt_mean - gt_std, gt_mean + gt_std,
                      color="black", alpha=0.15)
    # Prediction
    plt.plot(pred_mean, z, color="red", linestyle="--", linewidth=2, label="Pred")
    plt.fill_betweenx(z, pred_mean - pred_std, pred_mean + pred_std,
                      color="red", alpha=0.15)

    plt.gca().invert_yaxis()
    plt.xlabel("Value")
    plt.ylabel("Normalized depth")
    plt.legend()
    plt.title(sample_name)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{sample_name}_depth_profile.png", dpi=300)
    plt.close()

def plot_depth_residual(gt, pred, mask, sample_name, out_dir):
    depth = cartilage_depth(mask)
    z, res_mean, res_std = depth_profile(pred - gt, depth, mask)

    plt.figure(figsize=(4, 5))
    plt.plot(res_mean, z, color="blue", linewidth=2, label="Mean residual")
    plt.fill_betweenx(z, res_mean - res_std, res_mean + res_std,
                      color="blue", alpha=0.15, label="±1 std")
    plt.axvline(0, color="k", linestyle=":")
    plt.gca().invert_yaxis()
    plt.xlabel("Prediction - GT")
    plt.ylabel("Normalized depth")
    plt.legend()
    plt.title(f"Residual: {sample_name}")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{sample_name}_depth_residual.png", dpi=300)
    plt.close()

def tangential_profile(values, depth_map, mask, depth_range):
    d0, d1 = depth_range
    H, W = values.shape
    prof = np.full(W, np.nan)
    for c in range(W):
        sel = (
            (depth_map[:, c] >= d0) &
            (depth_map[:, c] <  d1) &
            mask[:, c] &
            np.isfinite(values[:, c])
        )
        if np.any(sel):
            prof[c] = np.mean(values[sel, c])
    return prof

def plot_tangential_profiles(gt, pred, mask, sample_name, out_dir,
                             depth_bands=((0.0, 0.2),
                                          (0.4, 0.6),
                                          (0.8, 1.0))):
    depth = cartilage_depth(mask)
    fig, axes = plt.subplots(len(depth_bands), 1, figsize=(6, 6), sharex=True)
    for ax, band in zip(axes, depth_bands):
        gt_p   = tangential_profile(gt,   depth, mask, band)
        pred_p = tangential_profile(pred, depth, mask, band)
        ax.plot(gt_p,   color="black", label="GT")
        ax.plot(pred_p, color="red", linestyle="--", label="Pred")
        ax.set_title(f"Depth band {band[0]:.1f}-{band[1]:.1f}")
        ax.set_ylabel("Value")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Column index (along surface)")
    fig.suptitle(f"Tangential profiles: {sample_name}")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/{sample_name}_tangential_profile.png", dpi=300)
    plt.close(fig)

def extract_and_save_profiles(gt, pred, mask, sample_name, out_dir):
    plot_depth_profiles(gt, pred, mask, sample_name, out_dir)
    plot_depth_residual(gt, pred, mask, sample_name, out_dir)
    plot_tangential_profiles(gt, pred, mask, sample_name, out_dir)


def plot_random_tiles(y_true, y_pred, m_out, sample_name, out_dir,
                      n_tiles=3, seed=0, show_diff=True):
    """
    Plot N random output patches as ONE FIGURE PER TILE.

    Each saved figure contains, side-by-side:
        [ Ground truth | Prediction | (optional) Error = Pred - GT ]

    Saved file:
        <sample>_tile<idx>_compare.png

    Parameters
    ----------
    y_true : (N, P, P) array  - ground-truth patches (unscaled)
    y_pred : (N, P, P) array  - predicted patches (unscaled)
    m_out  : (N, P, P) bool   - per-pixel cartilage mask for each patch
    sample_name : str
    out_dir : str
    n_tiles : int             - number of random patches to plot
    seed : int                - RNG seed for reproducibility
    show_diff : bool          - include the Pred - GT panel
    """
    if y_true.ndim != 3 or y_pred.ndim != 3:
        print(f"[plot_random_tiles] Skipped: expected 3D patch arrays, "
              f"got y_true.ndim={y_true.ndim}, y_pred.ndim={y_pred.ndim}")
        return

    N = y_true.shape[0]
    if N == 0:
        print(f"[plot_random_tiles] No patches available for {sample_name}")
        return

    # Prefer patches with a sufficient amount of cartilage
    if m_out is not None and m_out.ndim == 3:
        valid_frac = m_out.reshape(N, -1).mean(axis=1)
        candidates = np.where(valid_frac > 0.3)[0]
        if candidates.size < n_tiles:
            candidates = np.arange(N)
    else:
        candidates = np.arange(N)

    rng = np.random.default_rng(seed)
    n_tiles = min(n_tiles, candidates.size)
    chosen = rng.choice(candidates, size=n_tiles, replace=False)

    n_cols = 3 if show_diff else 2

    for idx in chosen:
        gt_p = y_true[idx].astype(np.float32).copy()
        pr_p = y_pred[idx].astype(np.float32).copy()
        mk_p = (m_out[idx].astype(bool)
                if (m_out is not None and m_out.ndim == 3)
                else np.ones_like(gt_p, dtype=bool))

        # Mask out non-cartilage pixels for display
        gt_disp = np.where(mk_p, gt_p, np.nan)
        pr_disp = np.where(mk_p, pr_p, np.nan)

        # Shared color range from GT (1-99 percentile of valid pixels)
        valid = mk_p & np.isfinite(gt_p) & np.isfinite(pr_p)
        if valid.any():
            vmin = float(np.percentile(gt_p[valid], 1))
            vmax = float(np.percentile(gt_p[valid], 99))
            if vmin == vmax:
                vmax = vmin + 1e-6
        else:
            vmin, vmax = None, None

        # Per-tile error stats (in the masked region)
        if valid.any():
            err = pr_p[valid] - gt_p[valid]
            tile_mae = float(np.mean(np.abs(err)))
            tile_rmse = float(np.sqrt(np.mean(err ** 2)))
        else:
            tile_mae = np.nan
            tile_rmse = np.nan

        fig, axes = plt.subplots(
            1, n_cols,
            figsize=(4.0 * n_cols, 4.2),
            squeeze=False,
        )
        axes = axes[0]  # row 0 only

        # --- GT ---
        im0 = axes[0].imshow(gt_disp, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0].set_title("Ground truth")
        axes[0].set_xticks([]); axes[0].set_yticks([])
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # --- Pred ---
        im1 = axes[1].imshow(pr_disp, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1].set_title("Prediction")
        axes[1].set_xticks([]); axes[1].set_yticks([])
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # --- Error (Pred - GT) ---
        if show_diff:
            diff = np.where(mk_p, pr_p - gt_p, np.nan)
            if np.isfinite(diff).any():
                vmax_d = float(np.nanpercentile(np.abs(diff), 99))
                if not np.isfinite(vmax_d) or vmax_d == 0:
                    vmax_d = 1e-6
            else:
                vmax_d = 1e-6
            im2 = axes[2].imshow(diff, cmap="coolwarm",
                                 vmin=-vmax_d, vmax=vmax_d)
            axes[2].set_title("Error (Pred - GT)")
            axes[2].set_xticks([]); axes[2].set_yticks([])
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        # Title with per-tile metrics
        suptitle = (f"{sample_name}  |  tile #{idx}  "
                    f"|  MAE={tile_mae:.3e}  RMSE={tile_rmse:.3e}")
        fig.suptitle(suptitle, y=1.02, fontsize=11)

        fig.tight_layout()
        save_path = os.path.join(out_dir,
                                 f"{sample_name}_tile{idx:05d}_compare.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot_random_tiles] Saved {save_path}")

# ============================================================
# Publication-style map visualization (shared with visualize_results.py)
# ============================================================



def map_metrics(gt, pred, mask=None, eps=1e-12):
    gt = np.asarray(gt, np.float64)
    pred = np.asarray(pred, np.float64)
    valid = np.isfinite(gt) & np.isfinite(pred)
    if mask is not None:
        valid &= mask.astype(bool)
    g = gt[valid].ravel()
    p = pred[valid].ravel()
    if g.size < 2:
        return dict(n=int(g.size), r2=np.nan, rmse=np.nan,
                    mae=np.nan, r=np.nan)
    err = p - g
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((g - g.mean()) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + eps)
    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))
    g0 = g - g.mean()
    p0 = p - p.mean()
    r = np.sum(g0 * p0) / (np.sqrt(np.sum(g0 ** 2) * np.sum(p0 ** 2)) + eps)
    return dict(n=int(g.size), r2=r2, rmse=rmse, mae=mae, r=r)


def _set_publication_style():
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "axes.linewidth": 0.8,
        "image.interpolation": "nearest",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })


def _copy_cmap_with_bad(cmap_name, bad_color="white"):
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color=bad_color)
    return cmap


def _robust_limits_from_pair(a, b, q_low=2, q_high=98):
    vals = np.concatenate([
        np.asarray(a)[np.isfinite(a)].ravel(),
        np.asarray(b)[np.isfinite(b)].ravel(),
    ])
    if vals.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(vals, q_low))
    vmax = float(np.nanpercentile(vals, q_high))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 0.0, 1.0
    if np.isclose(vmin, vmax):
        delta = max(abs(vmin) * 0.01, 1e-6)
        vmin -= delta
        vmax += delta
    return vmin, vmax


def _robust_error_limit(err, q=98):
    vals = np.asarray(err)[np.isfinite(err)]
    if vals.size == 0:
        return 1.0
    lim = float(np.nanpercentile(np.abs(vals), q))
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0
    return lim


def _add_compact_colorbar(fig, im, ax, label=None):
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015, shrink=0.88)
    cb.ax.tick_params(labelsize=9, length=3)
    if label:
        cb.set_label(label, fontsize=9)
    return cb


def plot_maps(map_gt, map_pred, title="", subtitle="",
              mask=None, save_path=None):
    _set_publication_style()

    map_gt = np.asarray(map_gt, dtype=np.float32)
    map_pred = np.asarray(map_pred, dtype=np.float32)

    if mask is not None:
        mask = mask.astype(bool)
        map_gt = np.where(mask, map_gt, np.nan)
        map_pred = np.where(mask, map_pred, np.nan)

    err = map_pred - map_gt
    m = map_metrics(map_gt, map_pred)

    def _fmt(x, fmt):
        return fmt.format(x) if x is not None and np.isfinite(x) else "NA"

    metrics_text = (
        f"$R^2$ = {_fmt(m['r2'], '{:.3f}')}   "
        f"$r$ = {_fmt(m['r'], '{:.3f}')}\n"
        f"RMSE = {_fmt(m['rmse'], '{:.3g}')}   "
        f"MAE = {_fmt(m['mae'], '{:.3g}')}   "
        f"$N$ = {m['n']}"
    )

    vmin, vmax = _robust_limits_from_pair(map_gt, map_pred,
                                          q_low=2, q_high=98)
    err_lim = _robust_error_limit(err, q=98)

    cmap_main = _copy_cmap_with_bad("viridis", bad_color="white")
    cmap_err = _copy_cmap_with_bad("coolwarm", bad_color="white")
    err_norm = TwoSlopeNorm(vmin=-err_lim, vcenter=0.0, vmax=err_lim)

    fig, axs = plt.subplots(3, 1, figsize=(8.0, 7.0),
                            constrained_layout=True)
    panel_titles = [
        "Ground truth",
        "Prediction",
        "Error: prediction − ground truth",
    ]

    images = []
    images.append(axs[0].imshow(map_gt, cmap=cmap_main,
                                vmin=vmin, vmax=vmax, aspect="equal"))
    images.append(axs[1].imshow(map_pred, cmap=cmap_main,
                                vmin=vmin, vmax=vmax, aspect="equal"))
    images.append(axs[2].imshow(err, cmap=cmap_err,
                                norm=err_norm, aspect="equal"))

    for ax, ptitle in zip(axs, panel_titles):
        ax.set_title(ptitle, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    _add_compact_colorbar(fig, images[0], axs[0])
    _add_compact_colorbar(fig, images[1], axs[1])
    _add_compact_colorbar(fig, images[2], axs[2])

    if title:
        fig.suptitle(str(title), y=1.04, fontsize=15, fontweight="bold")

    if subtitle:
        wrapped_subtitle = "\n".join(textwrap.wrap(str(subtitle), width=95))
        fig.text(0.5, 0.985, wrapped_subtitle,
                 ha="center", va="top", fontsize=11)

    axs[1].text(
        0.5, -0.12, metrics_text,
        transform=axs[1].transAxes,
        ha="center", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35",
                  facecolor="white", edgecolor="0.75", alpha=0.95),
    )

    axs[0].text(0.5, -0.07,
                f"Shared scale: {vmin:.3g} to {vmax:.3g}",
                transform=axs[0].transAxes,
                ha="center", va="top", fontsize=9, color="0.25")
    axs[2].text(0.5, -0.07,
                f"Error scale: ±{err_lim:.3g}",
                transform=axs[2].transAxes,
                ha="center", va="top", fontsize=9, color="0.25")

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        base, _ = os.path.splitext(save_path)
        fig.savefig(base + ".pdf", bbox_inches="tight")
        plt.close(fig)

    return m


def plot_loss_curves(history, out_dir, run_tag="", fold_tag="", logy=True):
    epochs   = np.asarray(history.get("epoch", []),      dtype=float)
    train_l  = np.asarray(history.get("train_loss", []), dtype=float)
    val_l    = np.asarray(history.get("val_loss", []),   dtype=float)
    val_sel  = np.asarray(history.get("val_select", []), dtype=float)
    lrs      = np.asarray(history.get("lr", []),         dtype=float)

    if epochs.size == 0:
        print("[plot_loss_curves] empty history; skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(epochs, train_l, label="Train loss (composite)", color="C0", lw=2)
    if np.isfinite(val_l).any():
        ax.plot(epochs, val_l, label="Val loss (composite)",
                color="C1", lw=2, linestyle="--")
    if np.isfinite(val_sel).any():
        ax.plot(epochs, val_sel, label="Val data-only (selection)",
                color="C3", lw=2, linestyle=":")

    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training curves | {fold_tag}", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")

    # Best-epoch marker on the SELECTION curve
    if np.isfinite(val_sel).any():
        best_ep_idx = int(np.nanargmin(val_sel))
        ax.axvline(epochs[best_ep_idx], color="red", linestyle=":", alpha=0.6)
        ax.scatter([epochs[best_ep_idx]], [val_sel[best_ep_idx]],
                   color="red", zorder=5)

    ax = axes[1]
    if np.isfinite(lrs).any():
        ax.plot(epochs, lrs, color="C3", lw=2)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_title("Learning rate schedule")
        ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(out_dir, "loss_curves.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curves saved to: {save_path}")

# ----------------- Memory-safe NaN/Inf sanitization ----------------- #
def sanitize_inplace(arr, name=""):
    """
    Replace NaN/Inf with 0 without allocating full-size boolean masks.
    Processes the array in chunks along axis 0 to keep memory bounded.
    """
    if arr is None or arr.size == 0:
        return arr

    # Cheap early-exit: if already clean, skip all work.
    # np.isfinite over a sample avoids the big allocation in the common case.
    try:
        # Fast scan: check chunk by chunk
        chunk_rows = max(1, 200_000_000 // max(1, arr[0].nbytes))
        all_clean = True
        for i in range(0, arr.shape[0], chunk_rows):
            sl = arr[i:i + chunk_rows]
            if not np.isfinite(sl).all():
                all_clean = False
                break
        if all_clean:
            return arr
    except Exception:
        pass

    # Sanitize in chunks, in place
    chunk_rows = max(1, 200_000_000 // max(1, arr[0].nbytes))
    for i in range(0, arr.shape[0], chunk_rows):
        sl = arr[i:i + chunk_rows]
        np.nan_to_num(sl, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  {name}: sanitized NaN/Inf in place (chunked)")
    return arr

# ----------------- Safe log1p for FunctionTransformer (picklable) ----------------- #
def _safe_log1p(y):
    """log1p with clipping to avoid log(<=0) from baseline residuals."""
    return np.log1p(np.clip(y, a_min=-1 + 1e-6, a_max=None))

# ----------------- Downsampling ----------------- #
def spatial_downsample_area(cube, mask, amide1, scale):
    if scale >= 1.0:
        return cube, mask, amide1
    H, W = cube.shape[:2]
    H_new, W_new = int(H * scale), int(W * scale)
    cube_ds = resize(cube, (H_new, W_new, cube.shape[2]),
                     order=1, anti_aliasing=True,
                     preserve_range=True).astype(np.float32)
    mask_ds = resize(mask.astype(np.float32), (H_new, W_new),
                     order=1, anti_aliasing=True,
                     preserve_range=True) > 0.5
    amide1_ds = resize(amide1, (H_new, W_new),
                       order=1, anti_aliasing=True,
                       preserve_range=True).astype(np.float32)
    return cube_ds, mask_ds, amide1_ds


def spatial_downsample(cube, mask, amide1, scale):
    if scale >= 1.0:
        return cube, mask, amide1
    cube_ds = ndimage_zoom(cube, (scale, scale, 1), order=1).astype(np.float32)
    mask_ds = ndimage_zoom(mask.astype(np.float32),
                           (scale, scale), order=0) > 0.5
    amide1_ds = ndimage_zoom(amide1, (scale, scale), order=1).astype(np.float32)
    return cube_ds, mask_ds, amide1_ds

# ----------------- Baseline correction ----------------- #

def baseline_offset(Y):
    """
    Offset baseline: subtract per-spectrum median.
    Y: (N, C)
    """
    med = np.median(Y, axis=1, keepdims=True)
    return Y - med

def emsc(Y, x=None, degree=2, ref_spectrum=None):
    """
    Vectorised EMSC. Same model as before:
      y ≈ a0 + a1*x + ... + ad*x^d + k*ref
    """
    Y = np.asarray(Y, dtype=np.float64)
    N, C = Y.shape
    if x is None:
        x = np.arange(C, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    xn = (x - x.min()) / (x.max() - x.min()) * 2 - 1

    if ref_spectrum is None:
        ref = Y.mean(axis=0)
    else:
        ref = np.asarray(ref_spectrum, dtype=np.float64)
        assert ref.shape[0] == C

    cols = [np.ones_like(xn)]
    for d in range(1, degree + 1):
        cols.append(xn ** d)
    cols.append(ref)
    M = np.vstack(cols).T   # (C, degree+2)

    # Solve all spectra in one shot:  Beta = (M^T M)^-1 M^T Y^T
    MtM_inv = np.linalg.inv(M.T @ M)        # (d+2, d+2)
    Beta = (MtM_inv @ (M.T @ Y.T)).T        # (N, d+2)

    k = Beta[:, -1:]                        # (N, 1)
    # Guard against divide-by-zero
    k = np.where(np.abs(k) < 1e-8, 1.0, k)

    # Polynomial baseline per spectrum: (N, C)
    baseline = Beta[:, :-1] @ M[:, :-1].T

    out = (Y - baseline) / k
    return out.astype(np.float32)

def apply_wavelength_exclusion(cube, wavelengths, exclude_ranges):
    """
    Remove spectral channels whose wavelength falls inside any of the
    given [low, high] ranges (inclusive).

    Parameters
    ----------
    cube : (H, W, C) float array
    wavelengths : (C,) float array, wavelength per channel in nm
    exclude_ranges : list of floats [low1, high1, low2, high2, ...] or None

    Returns
    -------
    cube_trimmed : (H, W, C_kept)
    wavelengths_trimmed : (C_kept,)
    keep_mask : (C,) bool, True where the channel is kept
    """
    if exclude_ranges is None or len(exclude_ranges) == 0:
        return cube, wavelengths, np.ones(wavelengths.shape[0], dtype=bool)

    if len(exclude_ranges) % 2 != 0:
        raise ValueError(
            f"--exclude_wl_ranges must contain an even number of values "
            f"(low1 high1 low2 high2 ...); got {len(exclude_ranges)}."
        )

    wavelengths = np.asarray(wavelengths, dtype=np.float64)

    # Verify that the wavelength axis matches the cube's spectral dim.
    if cube.ndim == 3 and cube.shape[2] != wavelengths.shape[0]:
        raise ValueError(
            f"apply_wavelength_exclusion: cube has {cube.shape[2]} "
            f"channels but wavelengths axis has {wavelengths.shape[0]} "
            f"entries. The exclusion appears to have been applied "
            f"upstream; aborting to prevent double application."
        )


    keep = np.ones(wavelengths.shape[0], dtype=bool)
    pairs = [(exclude_ranges[i], exclude_ranges[i + 1])
             for i in range(0, len(exclude_ranges), 2)]

    for lo, hi in pairs:
        if lo > hi:
            lo, hi = hi, lo
        in_range = (wavelengths >= lo) & (wavelengths <= hi)
        n_drop = int(in_range.sum())
        keep &= ~in_range
        print(f"[wl-exclude] Removing {n_drop} channels in "
              f"[{lo:.2f}, {hi:.2f}] nm.")

    n_kept = int(keep.sum())
    n_total = wavelengths.shape[0]
    print(f"[wl-exclude] Kept {n_kept}/{n_total} channels "
          f"({100 * n_kept / n_total:.1f}%).")

    if n_kept == 0:
        raise ValueError("All spectral channels excluded; nothing left.")

    return cube[:, :, keep], wavelengths[keep], keep

def apply_baseline_corr(cube, mask, mode="none", emsc_degree=2, wavelengths=None):
    """
    Baseline correction on full uHSI cube.
    cube: (H, W, C)
    mode: "none", "offset", "emsc"
    """
    if mode == "none":
        return cube.astype(np.float32)

    H, W, C = cube.shape
    # If the input is already float32 and contiguous, reshape is a view.
    if cube.dtype != np.float32:
        cube = cube.astype(np.float32, copy=False)
    X = cube.reshape(-1, C)  # view, not a copy
    mask_flat = np.asarray(mask).reshape(-1).astype(bool)

    if mode == "offset":
        # Modify in place; the caller already has a writable copy of uhsi
        if mask_flat.any():
            X[mask_flat] = baseline_offset(X[mask_flat])
        return X.reshape(H, W, C)
    elif mode == "emsc":
        if mask_flat.any():
            ref_spectrum = X[mask_flat].mean(axis=0)
            X[mask_flat] = emsc(
                X[mask_flat], x=wavelengths,
                degree=emsc_degree, ref_spectrum=ref_spectrum,
            )
        return X.reshape(H, W, C)
    else:
        raise ValueError(f"Unknown baseline_mode: {mode}")


# ----------------- Savitzky-Golay Smoothing ----------------- #

def apply_savgol_smoothing(X, window=7, polyorder=2, chunk_rows=1_000_000):
    X = np.asarray(X, dtype=np.float32)
    if X.size == 0:
        return X
    if window % 2 == 0:
        window += 1

    C = X.shape[-1]
    w = min(window, (C // 2) * 2 + 1)

    if X.ndim == 2:
        N = X.shape[0]
        if N <= chunk_rows:
            return savgol_filter(X, window_length=w, polyorder=polyorder,
                                 axis=-1).astype(np.float32)
        out = np.empty_like(X)
        for i in range(0, N, chunk_rows):
            end = min(i + chunk_rows, N)
            out[i:end] = savgol_filter(X[i:end], window_length=w,
                                        polyorder=polyorder, axis=-1)
        return out.astype(np.float32)

    elif X.ndim == 3:
        H, _, _ = X.shape
        out = np.empty_like(X)
        for h in range(H):
            out[h] = savgol_filter(X[h], window_length=w,
                                    polyorder=polyorder, axis=-1)
        return out.astype(np.float32)

    elif X.ndim == 4:
        N, P1, P2, C4 = X.shape
        pixels_per_sample = P1 * P2
        samples_per_chunk = max(1, chunk_rows // pixels_per_sample)
        out = np.empty_like(X)
        for i in range(0, N, samples_per_chunk):
            end = min(i + samples_per_chunk, N)
            block = X[i:end].reshape(-1, C4)
            block_sm = savgol_filter(block, window_length=w,
                                      polyorder=polyorder, axis=-1)
            out[i:end] = block_sm.reshape(end - i, P1, P2, C4)
        return out.astype(np.float32)


    else:
        raise ValueError(f"apply_savgol_smoothing: unsupported ndim={X.ndim}")

# ----------------- Dimensionality Reduction ----------------- #

def plsr_explained_variance(pls, X, Y):
    """
    Returns (var_x_ratio, var_y_ratio) as arrays of length n_components.
    X, Y must be the *same* data used for fit (centered internally by PLSR).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64).reshape(-1, 1) if Y.ndim == 1 else np.asarray(Y, dtype=np.float64)

    # Center as PLSRegression does internally
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)

    total_var_x = np.sum(Xc ** 2)
    total_var_y = np.sum(Yc ** 2)

    T = pls.x_scores_      # (n_samples, n_components)
    P = pls.x_loadings_    # (n_features, n_components)
    Q = pls.y_loadings_    # (n_targets,  n_components)

    var_x = np.array([np.sum((np.outer(T[:, i], P[:, i])) ** 2)
                      for i in range(T.shape[1])])
    var_y = np.array([np.sum((np.outer(T[:, i], Q[:, i])) ** 2)
                      for i in range(T.shape[1])])

    return var_x / total_var_x, var_y / total_var_y

def apply_dim_reduction(X_train, X_val, X_test,
                        mode="none", n_components=32,
                        y_train=None, m_train=None, m_in_train=None):
    """
    Dimensionality reduction over spectral/channel axis.

    Supports:
      - X with shape (N, C)
      - X with shape (N, P, P, C)
    """
    if mode == "none":
        return X_train, X_val, X_test, None

    # ---- reshape helpers ----
    def flatten_for_dr(X):
        if X is None or X.size == 0:
            return None, None
        if X.ndim == 2:
            # (N, C)
            return X, X.shape
        elif X.ndim == 4:
            # (N, P, P, C)
            N, P1, P2, C = X.shape
            return X.reshape(-1, C), (N, P1, P2, C)
        else:
            raise ValueError(f"Unsupported X.ndim={X.ndim} for dim reduction")

    def restore_shape(Z, shape):
        if Z is None or shape is None:
            return None
        if len(shape) == 2:
            N, _ = shape
            return Z.reshape(N, -1)
        elif len(shape) == 4:
            N, P1, P2, _ = shape
            return Z.reshape(N, P1, P2, -1)
        else:
            raise ValueError(f"Unsupported shape={shape} for restore_shape")

    Xtr_flat, shape_tr = flatten_for_dr(X_train)
    Xva_flat, shape_va = flatten_for_dr(X_val)
    Xte_flat, shape_te = flatten_for_dr(X_test)

    if Xtr_flat is None:
        return X_train, X_val, X_test, None

    # ---- choose DR model ----
    if mode == "pca":
        # Fit PCA only on cartilage pixels when a mask is available
        if m_train is not None and m_train.ndim == 3:
            if Xtr_flat.shape[0] == int(np.prod(m_train.shape)):
                m_fit = m_train.reshape(-1).astype(bool)
            else:
                m_fit = np.ones(Xtr_flat.shape[0], dtype=bool)
        else:
            m_fit = np.ones(Xtr_flat.shape[0], dtype=bool)

        n_used = int(m_fit.sum())
        print(f"[PCA] fitting on {n_used:,} / {Xtr_flat.shape[0]:,} "
              f"cartilage pixels ({100 * n_used / max(Xtr_flat.shape[0], 1):.1f}%)")

        dr = PCA(n_components=n_components)
        X_fit_pca = Xtr_flat[m_fit] if n_used >= max(n_components + 10, 100) \
            else Xtr_flat

        # Sanitize: drop non-finite rows and clip extreme values that overflow
        # during covariance computation (X.T @ X). Clipping at the 0.1/99.9
        # percentile is a standard robust preprocessing step before PCA.
        finite_rows = np.isfinite(X_fit_pca).all(axis=1)
        X_fit_pca = X_fit_pca[finite_rows]
        if X_fit_pca.shape[0] == 0:
            raise RuntimeError("PCA fit failed: no finite rows in cartilage pixels.")

        lo = np.nanpercentile(X_fit_pca, 0.1, axis=0)
        hi = np.nanpercentile(X_fit_pca, 99.9, axis=0)
        X_fit_pca = np.clip(X_fit_pca, lo, hi).astype(np.float64)

        dr.fit(X_fit_pca)

    elif mode == "plsr_dr":
        if y_train is None:
            raise ValueError("y_train is required for PLSR-based dim reduction")

        if X_train.ndim == 4:
            N, P1, P2, C = X_train.shape
            reps = P1 * P2

            # ---- NEW: collapse spectral target to per-pixel scalar for PLSR-DR ----
            # Spectra mode produces y_train with shape (N, P, P, C_ftir).
            # PLSR-DR needs one scalar per spectrum, so we average across the
            # spectral channel and fall through to the ndim==3 branch.
            if y_train.ndim == 4:
                print(f"[PLSR-DR] y_train is 4D (N,P,P,C_out={y_train.shape[-1]}); "
                    f"collapsing spectral axis with nanmean for DR fit.")
                y_train = np.nanmean(y_train, axis=-1).astype(np.float32)


            # ---- Build a per-pixel target for each spectrum in each patch ----
            if y_train.ndim == 3:
                # (N, P_out, P_out) per-pixel target map per patch.
                # We need it aligned to the input grid (P1, P2). If P_out < P_in,
                # center-pad with NaN so masked pixels are excluded later.
                P_out_h, P_out_w = y_train.shape[1], y_train.shape[2]
                if (P_out_h, P_out_w) == (P1, P2):
                    y_pix = y_train.astype(np.float32)
                elif P_out_h <= P1 and P_out_w <= P2:
                    y_pix = np.full((N, P1, P2), np.nan, dtype=np.float32)
                    h0 = (P1 - P_out_h) // 2
                    w0 = (P2 - P_out_w) // 2
                    y_pix[:, h0:h0 + P_out_h, w0:w0 + P_out_w] = y_train
                else:
                    raise ValueError(
                        f"y_train patch shape ({P_out_h},{P_out_w}) larger than "
                        f"X patch ({P1},{P2}); cannot align for PLSR-DR."
                    )
                y_rep = y_pix.reshape(-1)            # one target per spectrum

            elif y_train.ndim == 2:                  # (N, P_out*P_out) flat target
                P_side = int(round(np.sqrt(y_train.shape[1])))
                y_pix = y_train.reshape(N, P_side, P_side)
                if P_side == P1 and P_side == P2:
                    y_rep = y_pix.reshape(-1)
                else:
                    # center-pad to (P1, P2)
                    y_full = np.full((N, P1, P2), np.nan, dtype=np.float32)
                    h0 = (P1 - P_side) // 2
                    w0 = (P2 - P_side) // 2
                    y_full[:, h0:h0 + P_side, w0:w0 + P_side] = y_pix
                    y_rep = y_full.reshape(-1)

            elif y_train.ndim == 1:
                # Scalar target per patch — fall back to old behavior
                y_scalar = np.nan_to_num(y_train, nan=0.0,
                                        posinf=0.0, neginf=0.0)
                y_rep = np.repeat(y_scalar, reps)
            else:
                raise ValueError(f"Unsupported y_train.ndim={y_train.ndim}")

            # ---- Optional cartilage mask: drop background pixels ----
            if m_train is not None and m_train.ndim == 3:
                P_out_h, P_out_w = m_train.shape[1], m_train.shape[2]
                if (P_out_h, P_out_w) == (P1, P2):
                    m_pix = m_train.astype(bool)
                elif P_out_h <= P1 and P_out_w <= P2:
                    m_pix = np.zeros((N, P1, P2), dtype=bool)
                    h0 = (P1 - P_out_h) // 2
                    w0 = (P2 - P_out_w) // 2
                    m_pix[:, h0:h0 + P_out_h, w0:w0 + P_out_w] = m_train
                else:
                    m_pix = np.ones((N, P1, P2), dtype=bool)
                m_rep = m_pix.reshape(-1)
            else:
                m_rep = np.ones_like(y_rep, dtype=bool)

            # Drop NaN/Inf and non-cartilage pixels before fitting
            valid = m_rep & np.isfinite(y_rep) & np.isfinite(Xtr_flat).all(axis=1)
            Xfit = Xtr_flat[valid]
            yfit = y_rep[valid].reshape(-1, 1)

            # Subsample to keep PLSR fitting tractable (it scales poorly)
            max_pix = 500_000
            if Xfit.shape[0] > max_pix:
                # Use a deterministic but caller-controlled seed if possible.
                # Falls back to 0 when called from contexts without args.
                rng_seed = getattr(apply_dim_reduction, "_seed", 0)
                idx = np.random.default_rng(rng_seed).choice(
                    Xfit.shape[0], max_pix, replace=False
                )
                Xfit = Xfit[idx]
                yfit = yfit[idx]


            assert Xfit.shape[0] == yfit.shape[0], (
                f"PLSR-DR shape mismatch after masking: "
                f"X={Xfit.shape[0]} y={yfit.shape[0]}"
            )
            print(f"[PLSR-DR] fitting on {Xfit.shape[0]:,} pixels "
                f"(of {Xtr_flat.shape[0]:,})")

            dr = PLSRegression(n_components=n_components, scale=False)
            dr.fit(Xfit, yfit)

            # For variance reporting later, keep what was actually fit on:
            Xtr_for_report = Xfit
            y_for_report   = yfit.ravel()

        else:
            # X is (N, C); collapse y to scalar per sample if needed
            if y_train.ndim == 1:
                y_rep = y_train
            elif y_train.ndim == 2:
                y_rep = np.nanmean(y_train, axis=1)
            elif y_train.ndim == 3:
                y_rep = np.nanmean(y_train, axis=(1, 2))
            else:
                raise ValueError(f"Unsupported y_train.ndim={y_train.ndim}")
            y_rep = np.nan_to_num(y_rep, nan=0.0, posinf=0.0, neginf=0.0)

            dr = PLSRegression(n_components=n_components, scale=False)
            dr.fit(Xtr_flat, y_rep.reshape(-1, 1))

            Xtr_for_report = Xtr_flat
            y_for_report   = y_rep

    else:
        raise ValueError(f"Unknown dr_mode: {mode}")

    # ---- transform and reshape ----
    X_train_dr = restore_shape(dr.transform(Xtr_flat), shape_tr)
    X_val_dr = restore_shape(dr.transform(Xva_flat), shape_va) if Xva_flat is not None else X_val
    X_test_dr = restore_shape(dr.transform(Xte_flat), shape_te) if Xte_flat is not None else X_test

    # ---- report explained variance ----
    if mode == "pca":
        ratios = dr.explained_variance_ratio_
        print(f"[PCA] explained variance per component (%):")
        for i, r in enumerate(ratios, 1):
            print(f"  PC{i:02d}: {100*r:6.3f}%  (cum: {100*ratios[:i].sum():6.3f}%)")
        print(f"[PCA] total: {100*ratios.sum():.2f}% "
              f"with {n_components} components")

    elif mode == "plsr_dr":
        vx, vy = plsr_explained_variance(dr, Xtr_for_report, y_for_report)
        print(f"[PLSR-DR] explained variance per component (%):")
        print(f"{'Comp':>6} {'X (%)':>8} {'X cum':>8} {'Y (%)':>8} {'Y cum':>8}")
        for i, (ex, ey) in enumerate(zip(vx, vy), 1):
            print(f"{i:>6d} {100*ex:>8.3f} {100*vx[:i].sum():>8.3f} "
                  f"{100*ey:>8.3f} {100*vy[:i].sum():>8.3f}")
        print(f"[PLSR-DR] total X: {100*vx.sum():.2f}%, "
              f"total Y: {100*vy.sum():.2f}%")

    return X_train_dr, X_val_dr, X_test_dr, dr


# ----------------- Derivatives (Sav-Gol) ----------------- #

def apply_derivatives(X, der_mode="raw", sg_window=7,
                     polyorder=2, delta=1.0, chunk_rows=1_000_000):
    """
    Savitzky-Golay derivatives. Uses a SINGLE window length (`sg_window`)
    and polynomial order (`polyorder`) for both d/dx and d^2/dx^2, so
    the derivative settings are consistent with the smoothing settings
    selected by BO (--sg_window, --sg_poly).
    """
    if der_mode == "raw":
        return X

    X = np.asarray(X, dtype=np.float32)
    orig_shape = X.shape

    if X.ndim == 2:
        is_4d = False
        X_flat = X
    elif X.ndim == 4:
        is_4d = True
        N, P1, P2, C = orig_shape
        X_flat = X.reshape(-1, C)
    else:
        raise ValueError(f"Unsupported X.ndim={X.ndim}")

    C = X_flat.shape[-1]
    n_rows = X_flat.shape[0]
    # Force odd window and clip to channel count
    if sg_window % 2 == 0:
        sg_window += 1
    w = min(sg_window, C // 2 * 2 + 1)
    # Polynomial order must be < window
    p = min(polyorder, w - 1)

    if der_mode in ("der1", "der2"):
        C_out = C
    elif der_mode in ("raw+der1", "raw+der2", "der1+der2"):
        C_out = C * 2
    elif der_mode == "raw+der1+der2":
        C_out = C * 3
    else:
        raise ValueError(f"Unknown der_mode: {der_mode}")

    out_flat = np.empty((n_rows, C_out), dtype=np.float32)

    for i in range(0, n_rows, chunk_rows):
        end = min(i + chunk_rows, n_rows)
        chunk = X_flat[i:end]

        if der_mode == "der1":
            out_flat[i:end] = savgol_filter(
                chunk, window_length=w, polyorder=p,
                deriv=1, delta=delta, axis=-1)
        elif der_mode == "der2":
            out_flat[i:end] = savgol_filter(
                chunk, window_length=w, polyorder=p,
                deriv=2, delta=delta, axis=-1)
        elif der_mode == "raw+der1":
            d1 = savgol_filter(chunk, window_length=w, polyorder=p,
                               deriv=1, delta=delta, axis=-1)
            out_flat[i:end] = np.concatenate([chunk, d1], axis=-1)
            del d1
        elif der_mode == "raw+der2":
            d2 = savgol_filter(chunk, window_length=w, polyorder=p,
                               deriv=2, delta=delta, axis=-1)
            out_flat[i:end] = np.concatenate([chunk, d2], axis=-1)
            del d2
        elif der_mode == "raw+der1+der2":
            d1 = savgol_filter(chunk, window_length=w, polyorder=p,
                               deriv=1, delta=delta, axis=-1)
            d2 = savgol_filter(chunk, window_length=w, polyorder=p,
                               deriv=2, delta=delta, axis=-1)
            out_flat[i:end] = np.concatenate([chunk, d1, d2], axis=-1)
            del d1, d2
        elif der_mode == "der1+der2":
            d1 = savgol_filter(chunk, window_length=w, polyorder=p,
                               deriv=1, delta=delta, axis=-1)
            d2 = savgol_filter(chunk, window_length=w, polyorder=p,
                               deriv=2, delta=delta, axis=-1)
            out_flat[i:end] = np.concatenate([d1, d2], axis=-1)
            del d1, d2

    if is_4d:
        return out_flat.reshape(N, P1, P2, C_out).astype(np.float32)
    else:
        return out_flat.astype(np.float32)

# ----------------- Scaler ----------------- #

def make_sklearn_scaler(kind):
    if kind == "standard":
        return StandardScaler()
    elif kind == "minmax":
        return MinMaxScaler()
    elif kind == "robust":
        return RobustScaler()
    elif kind == "maxabs":
        return MaxAbsScaler()
    elif kind == "power":
        return PowerTransformer(method="yeo-johnson")
    elif kind == "quantile":
        return QuantileTransformer(output_distribution="normal", subsample=100000)
    elif kind == "log":
        return FunctionTransformer(
            func=_safe_log1p,
            inverse_func=np.expm1,
            validate=True,
            check_inverse=False,
        )
    elif kind == "log_standard":
        return Pipeline([
            ("log", FunctionTransformer(
                func=_safe_log1p,
                inverse_func=np.expm1,
                validate=True, check_inverse=False,
            )),
            ("scale", StandardScaler()),
        ])
    else:
        raise ValueError(f"Unknown scaler kind: {kind}")

# ----------------- Data extraction ----------------- #

def make_cartilage_mask(amide1, thresh=4.0, min_size=200):
    mask = amide1 > thresh
    mask = remove_small_objects(mask, min_size=min_size)
    return mask.astype(bool)

def extract_samples(cube, gt, mask, col_range,
                    overlap=0.0,
                    in_patch=1, in_avg_mode="avg",
                    out_patch=1, out_avg_mode="avg",
                    min_in_mask_frac=1.0,
                    min_out_mask_frac=1.0):
    """
    Supports gt of shape (H, W) (map mode) or (H, W, C_out) (spectra mode).
    Always returns m_out as the 2D spatial mask per sample.
    """
    H, W, C = cube.shape
    x0, x1 = col_range
    x0 = max(0, x0); x1 = min(W, x1)

    P_in = max(1, in_patch); P_out = max(1, out_patch)
    if P_in % 2 == 0 or P_out % 2 == 0:
        raise ValueError("in_patch and out_patch must be odd")
    rad_in = P_in // 2; rad_out = P_out // 2
    margin = max(rad_in, rad_out)

    if overlap < 0 or overlap >= 1:
        step = 1
    else:
        P_ref = max(P_in, P_out)
        step = max(1, int(round(P_ref * (1.0 - overlap))))

    gt = np.asarray(gt)
    has_spec = (gt.ndim == 3)

    X_list, y_list, m_list, m_in_list, coords = [], [], [], [], []

    for r in tqdm(range(margin, H - margin, step), leave=False):
        for c in range(max(margin, x0), min(W - margin, x1), step):
            r0_in, r1_in = r - rad_in, r + rad_in + 1
            c0_in, c1_in = c - rad_in, c + rad_in + 1
            patch_cube = cube[r0_in:r1_in, c0_in:c1_in, :]
            patch_mask_in = mask[r0_in:r1_in, c0_in:c1_in]

            r0_out, r1_out = r - rad_out, r + rad_out + 1
            c0_out, c1_out = c - rad_out, c + rad_out + 1
            patch_mask_out = mask[r0_out:r1_out, c0_out:c1_out]
            if has_spec:
                patch_gt = gt[r0_out:r1_out, c0_out:c1_out, :]
            else:
                patch_gt = gt[r0_out:r1_out, c0_out:c1_out]

            in_mask_frac = patch_mask_in.mean()
            out_mask_frac = patch_mask_out.mean()

            if in_avg_mode == "avg":
                if not patch_mask_in.any() or in_mask_frac < min_in_mask_frac:
                    continue
            elif in_avg_mode in {"flat", "patch"}:
                if in_mask_frac < min_in_mask_frac:
                    continue
            else:
                raise ValueError(f"Unknown in_avg_mode: {in_avg_mode}")

            if out_avg_mode == "avg":
                if not patch_mask_out.any() or out_mask_frac < min_out_mask_frac:
                    continue
            elif out_avg_mode in {"flat", "patch"}:
                if out_mask_frac < min_out_mask_frac:
                    continue
            else:
                raise ValueError(f"Unknown out_avg_mode: {out_avg_mode}")

            # build X
            if in_avg_mode == "avg" or P_in == 1:
                X_list.append(patch_cube[patch_mask_in].mean(axis=0).astype(np.float32))
            elif in_avg_mode == "flat":
                X_list.append(patch_cube.reshape(-1).astype(np.float32))
            elif in_avg_mode == "patch":
                X_list.append(patch_cube.astype(np.float32))

            # build y
            if out_avg_mode == "avg":
                if has_spec:
                    pg_flat = patch_gt.reshape(-1, patch_gt.shape[-1])
                    mflat = patch_mask_out.reshape(-1)
                    if mflat.any():
                        y_val = pg_flat[mflat].mean(axis=0)
                    else:
                        y_val = gt[r, c, :].astype(np.float32)
                    y_list.append(y_val.astype(np.float32))
                else:
                    y_list.append(np.float32(patch_gt[patch_mask_out].mean()))
            elif out_avg_mode == "flat":
                y_list.append(patch_gt.reshape(-1).astype(np.float32))
            elif out_avg_mode == "patch":
                y_list.append(patch_gt.astype(np.float32))

            m_list.append(patch_mask_out.astype(np.bool_))
            m_in_list.append(patch_mask_in.astype(np.bool_))
            coords.append((r, c))

    if not X_list:
        if in_avg_mode == "patch":
            X_empty = np.empty((0, P_in, P_in, C), dtype=np.float32)
        elif in_avg_mode == "flat":
            X_empty = np.empty((0, P_in * P_in * C), dtype=np.float32)
        else:
            X_empty = np.empty((0, C), dtype=np.float32)

        if has_spec:
            C_out = gt.shape[2]
            if out_avg_mode == "patch":
                y_empty = np.empty((0, P_out, P_out, C_out), dtype=np.float32)
            elif out_avg_mode == "flat":
                y_empty = np.empty((0, P_out * P_out * C_out), dtype=np.float32)
            else:
                y_empty = np.empty((0, C_out), dtype=np.float32)
        else:
            if out_avg_mode == "patch":
                y_empty = np.empty((0, P_out, P_out), dtype=np.float32)
            elif out_avg_mode == "flat":
                y_empty = np.empty((0, P_out * P_out), dtype=np.float32)
            else:
                y_empty = np.empty((0,), dtype=np.float32)

        m_empty    = np.empty((0, P_out, P_out), dtype=np.bool_)
        m_in_empty = np.empty((0, P_in,  P_in),  dtype=np.bool_)
        return X_empty, y_empty, m_empty, m_in_empty, []


    X = np.stack(X_list, axis=0).astype(np.float32); del X_list
    y = np.stack(y_list, axis=0).astype(np.float32); del y_list
    m_out = np.stack(m_list, axis=0); del m_list
    m_in  = np.stack(m_in_list, axis=0); del m_in_list
    gc.collect()
    return X, y, m_out, m_in, coords


def filter_valid_with_min(X, y, m_out, m_in, coords):
    X_arr = np.asarray(X); y_arr = np.asarray(y)
    m_out_arr = np.asarray(m_out); m_in_arr = np.asarray(m_in)
    X_flat = X_arr.reshape(X_arr.shape[0], -1)
    y_flat = y_arr.reshape(y_arr.shape[0], -1)
    valid = np.isfinite(X_flat).all(axis=1) & np.isfinite(y_flat).all(axis=1)
    return (X_arr[valid], y_arr[valid], m_out_arr[valid], m_in_arr[valid],
            [coords[i] for i, ok in enumerate(valid) if ok])

def filter_valid(X, y, m_out, coords):
    """
    Remove rows with non-finite values in X or y.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    m_arr = np.asarray(m_out)

    if X_arr.shape[0] != y_arr.shape[0] or X_arr.shape[0] != m_arr.shape[0]:
        raise ValueError(
            f"Length mismatch: X={X_arr.shape[0]}, y={y_arr.shape[0]}, m={m_arr.shape[0]}"
        )


    X_flat = X_arr.reshape(X_arr.shape[0], -1)
    y_flat = y_arr.reshape(y_arr.shape[0], -1)

    valid = np.isfinite(X_flat).all(axis=1) & np.isfinite(y_flat).all(axis=1)

    X_clean = X_arr[valid]
    y_clean = y_arr[valid]
    m_clean = m_arr[valid]
    coords_clean = [coords[i] for i, ok in enumerate(valid) if ok]
    return X_clean, y_clean, m_clean, coords_clean

# ----------------- Dataset ----------------- #

class UHsiStreamingDataset(Dataset):
    """
    Memory-efficient streaming dataset.

    Stores raw (unprocessed) patches and applies the per-sample, per-patch
    preprocessing steps (SG smoothing, spectrum normalization, derivatives,
    x-scaler, aux concat, y-scaler) inside __getitem__.

    Workers run __getitem__ in separate processes, so preprocessing
    overlaps with GPU training and only batch-sized intermediates exist
    in RAM at any moment.

    NOTE: This is numerically equivalent to the eager whole-array path
    because all operations are per-spectrum or per-patch; nothing depends
    on neighbouring patches except the fitted scalers, which are passed
    in from the main process after being fit on the training set.
    """
    def __init__(self, X_raw, y_raw, m_out, X_aux,
                 x_scaler=None, y_scaler=None,
                 smooth_x=False, sg_window=7, sg_poly=2,
                 spectrum_norm="none",
                 der_mode="raw", delta_wl=1.0,
                 dr_model=None,
                 model_type="unet"):
        # X_raw: (N, P, P, C) float32, raw spectra after baseline correction
        # y_raw: scaled or unscaled targets — see below
        # We expect y to already be y-scaled in the main process, because
        # the y-scaler is fit on all training pixels at once and applying
        # it per-patch is identical (it's elementwise).
        self.X_raw = X_raw
        self.y = y_raw
        self.m_out = m_out
        self.X_aux = X_aux  # may be None or zero-channel

        self.x_scaler = x_scaler
        self.smooth_x = smooth_x
        self.sg_window = sg_window
        self.sg_poly = sg_poly
        self.spectrum_norm = spectrum_norm
        self.der_mode = der_mode
        self.delta_wl = delta_wl
        self.dr_model = dr_model
        self.model_type = model_type

    def __len__(self):
        return self.X_raw.shape[0]

    def _preprocess_one(self, x_raw, x_aux):
        """
        Per-patch preprocessing. The Savitzky–Golay smoothing step and the
        derivative computation share a single polynomial fit, so the
        'smoothed signal + derivatives' triple is mathematically consistent
        (a single window length, a single polynomial order).

        Order (applied independently per spectrum):
            1. SG fit (one pass), produces [raw_or_smoothed, der1, der2]
            channels as requested by der_mode.
            2. Spectrum normalization (SNV/L2/...).
            3. x_scaler.transform   (fit on TRAIN cartilage pixels).
            4. Dimensionality reduction (PCA/PLSR-DR; fit on TRAIN).
            5. Aux channel concatenation (depth, position, global,
            cartilage mask), which bypasses spectral preprocessing.
        """
        x = x_raw  # (P, P, C)
        P1, P2, C = x.shape

        # ---- Step 1: single SG pass that yields all requested views ----
        # Decide which views are needed.
        needs_smooth = self.smooth_x
        needs_der1   = self.der_mode in ("der1", "raw+der1",
                                        "raw+der1+der2", "der1+der2")
        needs_der2   = self.der_mode in ("der2", "raw+der2",
                                        "raw+der1+der2", "der1+der2")
        keeps_raw    = self.der_mode in ("raw", "raw+der1", "raw+der2",
                                        "raw+der1+der2")

        # Force odd window and clip to C
        w = self.sg_window
        if w % 2 == 0:
            w += 1
        w = min(w, C // 2 * 2 + 1)
        p = min(self.sg_poly, w - 1)

        x_flat = x.reshape(-1, C)

        # The "base" view is either the smoothed signal (if smooth_x and
        # raw is kept) or the raw signal (if smooth_x off). When neither
        # smoothing nor derivatives are requested, we skip SG entirely.
        parts = []
        # Build / reuse cached SG kernels per worker.
        # scipy.signal.savgol_coeffs returns the per-window coefficient vector
        # that, when convolved with the signal, gives the (deriv-th) derivative.
        # Caching this and convolving directly saves ~10x vs. savgol_filter,
        # which rebuilds the coefficient matrix on every call.
        from scipy.signal import savgol_coeffs
        if not hasattr(self, "_sg_cache") or self._sg_cache.get("key") != (w, p, self.delta_wl):
            half = w // 2
            cache = {"key": (w, p, self.delta_wl)}
            cache["c0"] = savgol_coeffs(w, p, deriv=0, delta=self.delta_wl).astype(np.float32)
            cache["c1"] = savgol_coeffs(w, p, deriv=1, delta=self.delta_wl).astype(np.float32)
            cache["c2"] = savgol_coeffs(w, p, deriv=2, delta=self.delta_wl).astype(np.float32)
            cache["half"] = half
            self._sg_cache = cache

        def _sg_apply(x_flat, coeffs, half):
            # x_flat: (N, C). Pad with edge values (numpy's name for "nearest"),
            # then a single sliding-window contraction. Mathematically identical
            # to scipy.signal.savgol_filter for interior points; boundary
            # treatment uses edge replication, which matches scipy's default
            # mode='interp' in the interior and differs only in the ±half
            # boundary region (which the patch crop and shift-tolerant loss
            # ignore anyway).
            C = x_flat.shape[1]
            padded = np.pad(x_flat, ((0, 0), (half, half)), mode="edge")
            sw = np.lib.stride_tricks.sliding_window_view(padded, window_shape=2 * half + 1, axis=1)
            # sw shape: (N, C, w). Contract last axis with the SG coefficients.
            return np.einsum("ncw,w->nc", sw[:, :C, :], coeffs).astype(np.float32)


        if keeps_raw:
            if needs_smooth:
                base = _sg_apply(x_flat, self._sg_cache["c0"], self._sg_cache["half"])
            else:
                base = x_flat.astype(np.float32)
            parts.append(base)

        if needs_der1:
            d1 = _sg_apply(x_flat, self._sg_cache["c1"], self._sg_cache["half"])
            parts.append(d1)

        if needs_der2:
            d2 = _sg_apply(x_flat, self._sg_cache["c2"], self._sg_cache["half"])
            parts.append(d2)


        if not parts:
            # der_mode == "raw" and smooth_x == False: no SG at all.
            x = x.astype(np.float32)
        else:
            x = np.concatenate(parts, axis=-1).reshape(P1, P2, -1)

        # ---- Step 2: per-spectrum normalization ----
        if self.spectrum_norm != "none":
            x = apply_spectrum_norm(x, mode=self.spectrum_norm)

        # ---- Step 3: x_scaler ----
        if self.x_scaler is not None:
            P1, P2, Cc = x.shape

            # Consistency check: the scaler must expect exactly Cc features.
            # A mismatch indicates that the fit-time and transform-time
            # preprocessing pipelines disagree (typically because SG
            # derivatives or spectrum_norm were applied in the wrong
            # order at fit time). Emit a NaN patch so upstream metrics
            # can detect the inconsistency rather than average corrupted
            # values.
            expected = getattr(self.x_scaler, "n_features_in_", None)
            if expected is not None and int(expected) != Cc:
                print(f"[stream worker] x_scaler channel mismatch: "
                      f"scaler expects {expected}, got {Cc}. "
                      f"Emitting NaN patch. Check the SG / spectrum_norm / "
                      f"x_scaler ordering in run_one_fold.")
                z = np.full((P1 * P2, Cc), np.nan, dtype=np.float32)
                x = z.reshape(P1, P2, Cc).astype(np.float32)
            else:
                # Per-worker cache: extract affine parameters from sklearn
                # scalers so we can compute the transform with a single
                # broadcasted subtract+divide. This skips sklearn's per-call
                # validation/dispatch overhead. Non-affine scalers
                # (Power, Quantile) fall through to sklearn.
                if not hasattr(self, "_xs_cache"):
                    from sklearn.preprocessing import (
                        StandardScaler as _SS, MinMaxScaler as _MM,
                        RobustScaler as _RS, MaxAbsScaler as _MA,
                    )
                    s = self.x_scaler
                    cache = {"kind": "sklearn"}
                    try:
                        if isinstance(s, _SS):
                            cache = {
                                "kind": "affine",
                                "shift": s.mean_.astype(np.float32) if s.with_mean else np.zeros(Cc, dtype=np.float32),
                                "scale": s.scale_.astype(np.float32) if s.with_std else np.ones(Cc, dtype=np.float32),
                            }
                        elif isinstance(s, _MM):
                            # X_scaled = X * scale_ + min_
                            cache = {
                                "kind": "linear",
                                "scale": s.scale_.astype(np.float32),
                                "shift_add": s.min_.astype(np.float32),
                            }
                        elif isinstance(s, _RS):
                            cache = {
                                "kind": "affine",
                                "shift": s.center_.astype(np.float32) if s.with_centering else np.zeros(Cc, dtype=np.float32),
                                "scale": s.scale_.astype(np.float32) if s.with_scaling else np.ones(Cc, dtype=np.float32),
                            }
                        elif isinstance(s, _MA):
                            cache = {
                                "kind": "affine",
                                "shift": np.zeros(Cc, dtype=np.float32),
                                "scale": s.scale_.astype(np.float32),
                            }
                    except Exception:
                        cache = {"kind": "sklearn"}
                    self._xs_cache = cache

                try:
                    x_flat = x.reshape(-1, Cc)
                    c = self._xs_cache
                    if c["kind"] == "affine":
                        z = (x_flat - c["shift"]) / c["scale"]
                    elif c["kind"] == "linear":
                        z = x_flat * c["scale"] + c["shift_add"]
                    else:
                        z = self.x_scaler.transform(x_flat)
                    if not np.isfinite(z).all():
                        raise FloatingPointError(
                            "x_scaler.transform produced non-finite values"
                        )
                    z = np.clip(z, -1e6, 1e6).astype(np.float32)
                except Exception as e:
                    print(f"[stream worker] x_scaler.transform failed "
                        f"({type(e).__name__}: {e}); emitting NaN patch.")
                    z = np.full((P1 * P2, Cc), np.nan, dtype=np.float32)
                x = z.reshape(P1, P2, Cc).astype(np.float32)


        # ---- Step 4: dimensionality reduction ----
        if self.dr_model is not None:
            P1, P2, Cc = x.shape
            # Cache PCA components/mean as raw float32 arrays per worker, so we
            # avoid sklearn's Python dispatch on every patch. Same numerics as
            # self.dr_model.transform(), but ~3-5x faster.
            if not hasattr(self, "_dr_cache"):
                try:
                    from sklearn.decomposition import PCA as _PCA
                    if isinstance(self.dr_model, _PCA):
                        self._dr_cache = {
                            "kind": "pca",
                            "components": self.dr_model.components_.astype(np.float32),
                            "mean": (self.dr_model.mean_.astype(np.float32)
                                    if self.dr_model.mean_ is not None
                                    else np.zeros(Cc, dtype=np.float32)),
                            "whiten": bool(getattr(self.dr_model, "whiten", False)),
                            "explained_var": self.dr_model.explained_variance_.astype(np.float32)
                                            if getattr(self.dr_model, "whiten", False)
                                            else None,
                        }
                    else:
                        self._dr_cache = {"kind": "sklearn"}
                except Exception:
                    self._dr_cache = {"kind": "sklearn"}

            try:
                x_flat = x.reshape(-1, Cc)
                if self._dr_cache["kind"] == "pca":
                    z = (x_flat - self._dr_cache["mean"]) @ self._dr_cache["components"].T
                    if self._dr_cache["whiten"]:
                        z /= np.sqrt(self._dr_cache["explained_var"] + 1e-12)
                else:
                    z = self.dr_model.transform(x_flat)
                if not np.isfinite(z).all():
                    raise FloatingPointError(
                        "dr_model.transform produced non-finite values"
                    )
            except Exception as e:
                print(f"[stream worker] dr_model.transform failed "
                    f"({type(e).__name__}: {e}); emitting NaN patch.")
                z = np.full((P1 * P2, 1), np.nan, dtype=np.float32)
            x = z.reshape(P1, P2, -1).astype(np.float32)


        # ---- Step 5: aux channels ----
        if x_aux is not None and x_aux.size > 0 and x_aux.shape[-1] > 0:
            x = np.concatenate([x, x_aux.astype(np.float32)], axis=-1)

        if not np.isfinite(x).all():
            np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return x

    def __getitem__(self, idx):
        x_raw = self.X_raw[idx]
        x_aux = self.X_aux[idx] if self.X_aux is not None else None
        x = self._preprocess_one(x_raw, x_aux)

        # Channels-first for PyTorch conv layers
        if x.ndim == 3:
            x = np.transpose(x, (2, 0, 1))  # (C, P, P)

        y = self.y[idx]
        m = self.m_out[idx]

        # Mask broadcasting consistent with original UHsiDataset
        y_arr = np.asarray(y)
        m_arr = np.asarray(m)
        if y_arr.ndim == 4 - 1 and m_arr.ndim == 3:
            # Map mode, patch output: y is (P, P), m is (P, P)
            pass
        # (Other shapes: original UHsiDataset handles them at dataset-level;
        # here, since we stream one patch at a time, the only relevant case
        # for UNet/CNN is (P, P) y with (P, P) mask. PLSR/MLP paths still
        # use the eager UHsiDataset.)

        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(y_arr.astype(np.float32)),
            torch.from_numpy(m_arr.astype(np.bool_)),
        )

class UHsiDataset(Dataset):
    def __init__(self, X, y, m_out=None, model_type="mlp"):
        if X.ndim == 2:
            self.X = X.astype(np.float32)
        elif X.ndim == 4:
            self.X = np.transpose(X, (0, 3, 1, 2)).astype(np.float32)
        else:
            raise ValueError(
                f"UHsiDataset expects X.ndim in (2, 4); got {X.ndim}"
            )

        self.y = y.astype(np.float32)
        self.model_type = model_type

        if m_out is not None:
            m_out = np.asarray(m_out)

            if y.ndim == 1 and m_out.ndim == 3:
                # Scalar y per patch, 2D spatial mask -> any-pixel-cartilage flag
                self.m_out = m_out.any(axis=(1, 2)).astype(np.bool_)

            elif y.ndim == 2 and m_out.ndim == 3:
                # Two cases collapse here; disambiguate by trailing size:
                #   (a) map-mode flat:   y.shape == (N, P*P)
                #   (b) spectra-mode avg: y.shape == (N, C_out), C_out != P*P
                N, P1, P2 = m_out.shape
                if y.shape[0] == N and y.shape[1] == P1 * P2:
                    # (a) flat spatial map per patch
                    self.m_out = m_out.reshape(N, -1).astype(np.bool_)
                else:
                    # (b) spectra avg-output: valid if ANY cartilage pixel is in the patch
                    valid = m_out.any(axis=(1, 2))                    # (N,)
                    self.m_out = np.broadcast_to(
                        valid[:, None], y.shape
                    ).astype(np.bool_).copy()

            elif y.ndim == 4 and m_out.ndim == 3:
                # spectra patch: y is (N, P, P, C_out); broadcast spatial mask over channel
                self.m_out = np.broadcast_to(
                    m_out[..., None], y.shape
                ).astype(np.bool_).copy()

            elif y.ndim == 2 and m_out.ndim == 1:
                # spectra per-pixel: y is (N, C_out); mask is (N,)
                self.m_out = np.broadcast_to(
                    m_out[:, None], y.shape
                ).astype(np.bool_).copy()

            else:
                self.m_out = m_out.astype(np.bool_)

        else:
            if y.ndim == 1:
                self.m_out = np.ones((y.shape[0],), dtype=np.bool_)
            else:
                self.m_out = np.ones_like(y, dtype=np.bool_)


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx].astype(np.float32)),
            torch.from_numpy(np.asarray(self.y[idx], dtype=np.float32)),
            torch.from_numpy(np.asarray(self.m_out[idx], dtype=np.bool_)),
        )

# ----------------- Models ----------------- #

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128), dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.GroupNorm(num_groups=min(32, h), num_channels=h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(1)

class CNN1D(nn.Module):
    def __init__(self, in_ch, hidden=64, n_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        c = 1
        for i in range(n_layers):
            layers.append(nn.Conv1d(c, hidden, kernel_size=5, padding=2))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.GELU())
            if i >= 1:
                layers.append(nn.Dropout(dropout))
            c = hidden
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x).squeeze(2)
        return self.head(x).squeeze(1)


class CNN2D(nn.Module):
    def __init__(self, in_ch, patch, hidden=[32, 64], dropout=0.1):
        super().__init__()

        layers = []
        d = in_ch
        for i, h in enumerate(hidden):
            layers.append(nn.Conv2d(d, h, 3, padding=1))
            layers.append(nn.GELU())
            if i >= 2:
                layers.append(nn.Dropout2d(dropout))
            d = h

        
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        # x: (B, C, P, P)
        x = self.features(x).flatten(1)
        return self.head(x).squeeze(1)

# ============================================================
#  PatchGAN discriminator and adversarial / cycle losses
# ============================================================

class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN discriminator (Isola et al., Pix2Pix).
    Operates on the CONCATENATION of (input_uhsi_summary, target_or_pred)
    for conditional pix2pix, OR on a single image for unconditional CycleGAN.
    """
    def __init__(self, in_ch, base_ch=64, n_layers=3):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, base_ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base_ch
        for i in range(1, n_layers):
            ch_next = min(base_ch * (2 ** i), 512)
            layers += [
                nn.Conv2d(ch, ch_next, kernel_size=4, stride=2, padding=1,
                          bias=False),
                nn.GroupNorm(num_groups=min(32, ch_next), num_channels=ch_next),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = ch_next
        ch_next = min(ch * 2, 512)
        layers += [
            nn.Conv2d(ch, ch_next, kernel_size=4, stride=1, padding=1,
                      bias=False),
            nn.GroupNorm(num_groups=min(32, ch_next), num_channels=ch_next),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch_next, 1, kernel_size=4, stride=1, padding=1),
        ]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class ImageBuffer:
    """
    Shrivastava et al. (2017) replay buffer. Stores up to `pool_size`
    previously-generated fakes. With probability 0.5, returns a stored
    fake instead of the current one. Stabilises D training.
    """
    def __init__(self, pool_size=50):
        self.pool_size = int(pool_size)
        self.items = []

    def query(self, batch):
        if self.pool_size == 0:
            return batch
        out = []
        for img in batch:
            img = img.unsqueeze(0)
            if len(self.items) < self.pool_size:
                self.items.append(img.detach().clone())
                out.append(img)
            else:
                if torch.rand(1).item() < 0.5:
                    idx = torch.randint(0, self.pool_size, (1,)).item()
                    tmp = self.items[idx].clone()
                    self.items[idx] = img.detach().clone()
                    out.append(tmp)
                else:
                    out.append(img)
        return torch.cat(out, dim=0)


def adversarial_loss(pred, is_real, mode="lsgan"):
    """
    pred: discriminator logits (B, 1, h, w) or (B, 1).
    is_real: bool — target label.
    """
    if mode == "lsgan":
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return F.mse_loss(pred, target)
    elif mode == "vanilla":
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return F.binary_cross_entropy_with_logits(pred, target)
    elif mode == "hinge":
        if is_real:
            return F.relu(1.0 - pred).mean()
        else:
            return F.relu(1.0 + pred).mean()
    else:
        raise ValueError(f"Unknown gan_loss_type: {mode}")


def _to_4d_image(t, layout=None):
    """
    Convert a model output to (B, C, H, W).

    layout:
        "bchw"  : assume (B, C, H, W) — return as is.
        "bhwc"  : assume (B, H, W, C) — permute to (B, C, H, W).
        None    : 3D -> (B, 1, H, W); 4D -> error (caller must specify).

    Avoids ambiguity when out_patch == n_maps (e.g. (B, 5, 5, 5)).
    """
    if t.ndim == 3:
        return t.unsqueeze(1)
    if t.ndim == 4:
        if layout == "bchw":
            return t
        if layout == "bhwc":
            return t.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            "_to_4d_image: ambiguous 4D tensor; pass layout='bchw' or 'bhwc'."
        )
    raise ValueError(f"_to_4d_image: unsupported ndim={t.ndim}")




class SpectralProjector(nn.Module):
    """
    1x1 conv that projects the full uHSI cube (B, C_in, H, W) down to
    (B, n_proj, H, W) for conditioning the discriminator. Learned, so the
    discriminator sees a non-trivial low-dimensional summary of the input.
    """
    def __init__(self, in_ch, n_proj=8):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, n_proj * 2, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, n_proj * 2),
                         num_channels=n_proj * 2),
            nn.GELU(),
            nn.Conv2d(n_proj * 2, n_proj, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.proj(x)


def _summary_input_for_D(x_in, projector=None):
    """
    Build a conditioning tensor for the PatchGAN. If `projector` is given,
    use a learned 1x1 projection to n_proj channels; otherwise fall back
    to the channel mean (single channel).
    """
    if projector is not None:
        return projector(x_in)
    return x_in.mean(dim=1, keepdim=True)



class InverseGeneratorUNet(nn.Module):
    """
    G_B: FTIR target -> uHSI cube. Used only for cyclegan.
    Mirrors UNet2D but with configurable in/out channels and 'patch' output.
    """
    def __init__(self, in_ch, out_ch, base_ch=32, depth=3, dropout=0.1):
        super().__init__()
        self.depth = depth
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        c = in_ch
        enc_channels = []
        for i in range(depth):
            co = base_ch * (2 ** i)
            self.encoders.append(ResConvBlock(c, co, dropout, use_se=False))
            self.pools.append(
                nn.Conv2d(co, co, kernel_size=3, stride=2, padding=1, bias=False)
            )
            enc_channels.append(co)
            c = co
        self.bottleneck = ResConvBlock(c, base_ch * (2 ** depth), dropout,
                                       use_se=False)
        c = base_ch * (2 ** depth)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):
            cs = enc_channels[i]
            co = cs
            self.upconvs.append(nn.Sequential(
                nn.Conv2d(c, co * 4, kernel_size=3, padding=1, bias=False),
                nn.PixelShuffle(upscale_factor=2),
            ))
            self.decoders.append(
                ResConvBlock(co + cs, co, dropout, use_se=False)
            )
            c = co
        self.out_conv = nn.Conv2d(enc_channels[0], out_ch, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        factor = 2 ** self.depth
        H_pad = math.ceil(H / factor) * factor
        W_pad = math.ceil(W / factor) * factor
        x = F.pad(x, (0, W_pad - W, 0, H_pad - H), mode="reflect")
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x); skips.append(x); x = pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            th, tw = skip.shape[2], skip.shape[3]
            if x.shape[2] != th or x.shape[3] != tw:
                x = F.pad(x, (0, tw - x.shape[3], 0, th - x.shape[2]),
                          mode="reflect")
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        x = self.out_conv(x)
        return x[:, :, :H, :W]



class PLSRWrapper:
    def __init__(self, n_components=10):
        self.model = PLSRegression(n_components=n_components, scale=False)
        self.fitted = False

    def fit(self, X, y):
        self.model.fit(X, y.reshape(-1, 1))
        self.fitted = True

    def predict(self, X):
        return self.model.predict(X).ravel()
    

class ChannelSEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, max(ch // reduction, 4)),
            nn.GELU(),
            nn.Linear(max(ch // reduction, 4), ch),
            nn.Sigmoid(),
        )


    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"ChannelSEBlock expects 4D input (B,C,H,W), got ndim={x.ndim}")
        B, C, H, W = x.shape
        w = x.mean(dim=(2, 3))        # (B, C)
        w = self.fc(w).unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1)
        return x * w


def _gn(ch):
    # GroupNorm with a sensible default; groups capped at 32.
    return nn.GroupNorm(num_groups=min(32, ch), num_channels=ch)


class ResConvBlock(nn.Module):
    """Two-conv block with residual skip and optional SE attention."""
    def __init__(self, in_ch, out_ch, dropout=0.1, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = _gn(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = _gn(out_ch)
        self.drop = nn.Dropout2d(dropout)
        self.relu = nn.GELU()   # kept attribute name 'self.relu' to avoid changing forward()


        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                _gn(out_ch),
            )
            if in_ch != out_ch
            else nn.Identity()
        )

        self.se = ChannelSEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.relu(out + identity)
        return out


class PretrainedUNet(nn.Module):
    def __init__(self, in_ch, out_ch=1,
                 encoder_name="efficientnet-b3",
                 encoder_weights="imagenet",
                 out_mode="patch", out_patch=None, dropout=0.1,
                 target_mode="map"):
        super().__init__()
        if not _SMP_AVAILABLE:
            raise ImportError(
                "segmentation_models_pytorch is required for --model pretrained_unet. "
                "Install with: pip install segmentation-models-pytorch"
            )

        assert out_mode in ["avg", "flat", "patch"]
        self.out_mode = out_mode
        self.out_patch = out_patch
        self.target_mode = target_mode
        self.out_ch = out_ch

        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_ch,
            classes=out_ch,        # <-- was hard-coded to 1
            activation=None,
        )
        self.head_dropout = nn.Dropout2d(dropout)


    def forward(self, x):
        B, C, H, W = x.shape
        f = 32
        H_pad = math.ceil(H / f) * f
        W_pad = math.ceil(W / f) * f
        pad_w = W_pad - W
        pad_h = H_pad - H
        pad_mode = "reflect" if (pad_w < W and pad_h < H) else "replicate"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)

        out = self.net(x)                  # (B, out_ch, H_pad, W_pad)
        out = self.head_dropout(out)
        out = out[:, :, :H, :W]            # (B, out_ch, H, W)

        if self.out_patch is not None:
            P = self.out_patch
            h0 = (out.shape[2] - P) // 2
            w0 = (out.shape[3] - P) // 2
            out = out[:, :, h0:h0 + P, w0:w0 + P]

        if self.target_mode == "map":
            m = out.squeeze(1)             # (B, P, P)
            if self.out_mode == "patch":
                return m
            elif self.out_mode == "flat":
                return m.reshape(B, -1)
            else:  # avg
                return m.mean(dim=(1, 2))
        else:  # spectra
            # Match UNet2D_FTIR layout: (B, P, P, C_out)
            if self.out_mode == "patch":
                return out.permute(0, 2, 3, 1).contiguous()
            elif self.out_mode == "flat":
                return out.reshape(B, -1)
            else:  # avg
                return out.mean(dim=(2, 3))   # (B, C_out)



class UNet2D(nn.Module):
    """
    Improved UNet with:
      - Dynamic depth from args
      - Residual conv blocks with SE attention
      - Proper weight initialization
      - Deep supervision option
    
    Input:  (B, C_in, H, W)
    Output:
        out_mode == "patch": (B, H, W)
        out_mode == "flat":  (B, H*W)
        out_mode == "avg":   (B,)
    """
    def __init__(self, in_ch, base_ch=32, dropout=0.1, out_mode="patch",
                out_patch=None, depth=3, use_se=True,
                out_ch=1, target_mode="map"):
        """
        out_ch:
            Number of output channels per pixel.
            - target_mode='map'    -> out_ch must be 1
            - target_mode='spectra'-> out_ch = C_ftir
        target_mode:
            Controls the output tensor layout:
            - 'map'     : (B, P, P) (or flat/avg)
            - 'spectra' : (B, P, P, C_out) (or flat/avg)
        """
        super().__init__()                                   # FIRST

        assert out_mode in ["avg", "flat", "patch"]
        assert target_mode in ("map", "spectra")
        if target_mode == "map" and out_ch != 1:
            raise ValueError("UNet2D: target_mode='map' requires out_ch=1.")

        self.out_ch = int(out_ch)
        self.target_mode = target_mode
        self.out_mode = out_mode
        self.out_patch = out_patch
        self.depth = depth

        # ---- Encoder ----
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch_in = in_ch
        enc_channels = []
        for i in range(depth):
            ch_out = base_ch * (2 ** i)
            self.encoders.append(ResConvBlock(ch_in, ch_out, dropout, use_se=use_se))
            self.pools.append(
                nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=2, padding=1, bias=False)
            )
            enc_channels.append(ch_out)
            ch_in = ch_out

        # ---- Bottleneck ----
        bottleneck_ch = base_ch * (2 ** depth)
        self.bottleneck = ResConvBlock(ch_in, bottleneck_ch, dropout, use_se=use_se)

        # ---- Decoder ----
        self.upconvs = nn.ModuleList()
        self.size_adapts = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch_in = bottleneck_ch
        for i in reversed(range(depth)):
            ch_skip = enc_channels[i]
            ch_out = ch_skip  # decoder output matches encoder at same level
            self.upconvs.append(nn.Sequential(
                nn.Conv2d(ch_in, ch_out * 4, kernel_size=3, padding=1, bias=False),
                nn.PixelShuffle(upscale_factor=2),
            ))
            #  1x1 channel adapter applied after crop/pad
            self.size_adapts.append(
                nn.Conv2d(ch_out, ch_out, kernel_size=1, bias=False)
            )
            # After concat with skip: ch_out + ch_skip = 2 * ch_out
            self.decoders.append(
                ResConvBlock(ch_out + ch_skip, ch_out, dropout, use_se=use_se)
            )
            ch_in = ch_out

        # ---- Output head ----
        mid_ch = max(enc_channels[0] // 2, 1)
        self.head_dropout = nn.Dropout2d(dropout)
        self.out_conv = nn.Sequential(
            nn.Conv2d(enc_channels[0], mid_ch, 1),
            nn.GELU(),
            nn.Conv2d(mid_ch, self.out_ch, 1),  # multi-channel head
        )

        # ---- Weight initialization ----
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        B, C, H, W = x.shape

        # ---- Pad to multiple of 2^depth ----
        factor = 2 ** self.depth
        H_pad = math.ceil(H / factor) * factor
        W_pad = math.ceil(W / factor) * factor
        pad_h = H_pad - H
        pad_w = W_pad - W
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # ---- Encoder ----
        skip_features = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skip_features.append(x)
            x = pool(x)

        # ---- Bottleneck ----
        x = self.bottleneck(x)

        # ---- Decoder ----
        for upconv, adapt, dec, skip in zip(
            self.upconvs, self.size_adapts, self.decoders, reversed(skip_features)
        ):
            x = upconv(x)

            # ---- Size-match to skip via crop/pad (no interpolation, no blur) ----
            tgt_h, tgt_w = skip.shape[2], skip.shape[3]
            cur_h, cur_w = x.shape[2],   x.shape[3]

            # Crop if larger
            if cur_h > tgt_h:
                start = (cur_h - tgt_h) // 2
                x = x[:, :, start:start + tgt_h, :]
                cur_h = tgt_h
            if cur_w > tgt_w:
                start = (cur_w - tgt_w) // 2
                x = x[:, :, :, start:start + tgt_w]
                cur_w = tgt_w

            # Pad (reflect) if smaller
            if cur_h < tgt_h or cur_w < tgt_w:
                dec_pad_h = tgt_h - cur_h
                dec_pad_w = tgt_w - cur_w
                x = F.pad(x, (dec_pad_w // 2, dec_pad_w - dec_pad_w // 2,
                            dec_pad_h // 2, dec_pad_h - dec_pad_h // 2),
                        mode="reflect")


            # Learned channel-wise refinement (no spatial mixing, no blur)
            x = adapt(x)

            x = torch.cat([x, skip], dim=1)
            x = dec(x)


        x = self.head_dropout(x)
        out = self.out_conv(x)                # (B, out_ch, H_pad, W_pad)
        out = out[:, :, :H, :W]                # crop padding -> (B, out_ch, H, W)

        if self.out_patch is not None:
            P_out = self.out_patch
            cur_H, cur_W = out.shape[2], out.shape[3]
            if P_out > cur_H or P_out > cur_W:
                raise ValueError(
                    f"out_patch={P_out} larger than input H,W={cur_H},{cur_W}"
                )
            h0 = (cur_H - P_out) // 2
            w0 = (cur_W - P_out) // 2
            out = out[:, :, h0:h0 + P_out, w0:w0 + P_out]

        B = out.shape[0]
        if self.target_mode == "map":
            # Back-compat: (B, P, P) / (B, P*P) / (B,)
            m = out.squeeze(1)                 # (B, P, P)
            if self.out_mode == "patch":
                return m
            elif self.out_mode == "flat":
                return m.reshape(B, -1)
            else:  # "avg"
                return m.mean(dim=(1, 2))
        else:
            # Spectra: BHWC layout, identical to UNet2D_FTIR / MultiMapUNet2D
            if self.out_mode == "patch":
                return out.permute(0, 2, 3, 1).contiguous()   # (B, P, P, C_out)
            elif self.out_mode == "flat":
                return out.reshape(B, -1)
            else:  # "avg"
                return out.mean(dim=(2, 3))                    # (B, C_out)


class UNet2D_simple(nn.Module):
    """
    UNet-like model for spectral image regression on patches.

    Input:  (B, C_in, H, W)
    Output:
        out_mode == "patch": (B, H, W)
        out_mode == "flat":  (B, H*W)
        out_mode == "avg":   (B,)
    """
    def __init__(self, in_ch, base_ch=32, dropout=0.1, out_mode="patch", out_patch=None, depth=3):
        super().__init__()
        assert out_mode in ["avg", "flat", "patch"]
        self.out_mode = out_mode
        self.out_patch = out_patch
        self.depth = depth

        # Encoder
        self.enc1 = self._block(in_ch, base_ch, dropout)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = self._block(base_ch, base_ch * 2, dropout)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = self._block(base_ch * 2, base_ch * 4, dropout)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self._block(base_ch * 4, base_ch * 8, dropout)
        self.bottleneck_rep = self._block(base_ch * 8, base_ch * 8, dropout)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = self._block(base_ch * 8, base_ch * 4, dropout)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = self._block(base_ch * 4, base_ch * 2, dropout)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = self._block(base_ch * 2, base_ch, dropout)

        # Output: single channel per pixel
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

    def _block(self, in_ch, out_ch, dropout):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )


    def forward(self, x):
        B, C, H, W = x.shape

        # pad to multiple of 2**depth so pooling/upsampling is exact
        factor = 2 ** self.depth
        H_pad = math.ceil(H / factor) * factor
        W_pad = math.ceil(W / factor) * factor
        pad_h = H_pad - H
        pad_w = W_pad - W
        # pad (left, right, top, bottom) -> pad only on right and bottom
        x = F.pad(x, (0, pad_w, 0, pad_h))

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)
        for _ in range(2):
            residual = self.bottleneck_rep(b)
            b = b + 0.5 * residual  # scale residual to prevent magnitude growth

        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)           # (B, 1, H_pad, W_pad)
        out = out[:, :, :H, :W]           # crop back to original input size (P_in x P_in)
        patch_map = out.squeeze(1)        # (B, H, W)

        # If out_patch is specified and smaller, center-crop to (out_patch x out_patch)
        if self.out_patch is not None:
            P_out = self.out_patch
            if P_out > H or P_out > W:
                raise ValueError(f"out_patch={P_out} larger than input H,W={H},{W}")
            h0 = (H - P_out) // 2
            w0 = (W - P_out) // 2
            patch_map = patch_map[:, h0:h0 + P_out, w0:w0 + P_out]  # (B, P_out, P_out)

        if self.out_mode == "patch":
            return patch_map                           # (B, P_out, P_out) or (B, H, W)
        elif self.out_mode == "flat":
            return patch_map.view(B, -1)               # (B, P_out*P_out)
        else:  # "avg"
            return patch_map.mean(dim=(1, 2))          # (B,)

class MultiMapUNet2D(nn.Module):
    """
    Multi-map UNet: same architecture as UNet2D but outputs N channels,
    one per predicted map. Shared encoder/decoder; the final 1x1 conv
    produces n_maps channels simultaneously.

    Input:  (B, C_in, H, W)
    Output:
        out_mode == "patch": (B, n_maps, H, W)  -> permuted to (B, H, W, n_maps)
        out_mode == "flat":  (B, n_maps * H * W)
        out_mode == "avg":   (B, n_maps)
    """
    def __init__(self, in_ch, n_maps, base_ch=32, dropout=0.1,
                 out_mode="patch", out_patch=None, depth=3,
                 use_se=False, share_head=True):
        super().__init__()
        assert out_mode in ["avg", "flat", "patch"]
        self.out_mode = out_mode
        self.out_patch = out_patch
        self.depth = depth
        self.n_maps = n_maps
        self.share_head = share_head

        # ---- Encoder (identical to UNet2D) ----
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch_in = in_ch
        enc_channels = []
        for i in range(depth):
            ch_out = base_ch * (2 ** i)
            self.encoders.append(ResConvBlock(ch_in, ch_out, dropout, use_se=use_se))
            self.pools.append(
                nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=2, padding=1, bias=False)
            )
            enc_channels.append(ch_out)
            ch_in = ch_out

        bottleneck_ch = base_ch * (2 ** depth)
        self.bottleneck = ResConvBlock(ch_in, bottleneck_ch, dropout, use_se=use_se)

        # ---- Decoder ----
        self.upconvs = nn.ModuleList()
        self.size_adapts = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch_in = bottleneck_ch
        for i in reversed(range(depth)):
            ch_skip = enc_channels[i]
            ch_out = ch_skip
            self.upconvs.append(nn.Sequential(
                nn.Conv2d(ch_in, ch_out * 4, kernel_size=3, padding=1, bias=False),
                nn.PixelShuffle(upscale_factor=2),
            ))
            self.size_adapts.append(
                nn.Conv2d(ch_out, ch_out, kernel_size=1, bias=False)
            )
            self.decoders.append(
                ResConvBlock(ch_out + ch_skip, ch_out, dropout, use_se=use_se)
            )
            ch_in = ch_out

        # ---- Output head: multi-channel ----
        mid_ch = max(enc_channels[0] // 2, 1)
        self.head_dropout = nn.Dropout2d(dropout)
        if share_head:
            # Single shared head producing n_maps channels
            self.out_conv = nn.Sequential(
                nn.Conv2d(enc_channels[0], mid_ch, 1),
                nn.GELU(),
                nn.Conv2d(mid_ch, n_maps, 1),
            )
        else:
            # Separate 1x1 conv heads, concatenated
            self.shared_proj = nn.Sequential(
                nn.Conv2d(enc_channels[0], mid_ch, 1),
                nn.GELU(),
            )
            self.heads = nn.ModuleList([
                nn.Conv2d(mid_ch, 1, 1) for _ in range(n_maps)
            ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B, C, H, W = x.shape

        factor = 2 ** self.depth
        H_pad = math.ceil(H / factor) * factor
        W_pad = math.ceil(W / factor) * factor
        pad_h = H_pad - H
        pad_w = W_pad - W
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        skip_features = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skip_features.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upconv, adapt, dec, skip in zip(
            self.upconvs, self.size_adapts, self.decoders, reversed(skip_features)
        ):
            x = upconv(x)
            tgt_h, tgt_w = skip.shape[2], skip.shape[3]
            cur_h, cur_w = x.shape[2], x.shape[3]
            if cur_h > tgt_h:
                start = (cur_h - tgt_h) // 2
                x = x[:, :, start:start + tgt_h, :]
                cur_h = tgt_h
            if cur_w > tgt_w:
                start = (cur_w - tgt_w) // 2
                x = x[:, :, :, start:start + tgt_w]
                cur_w = tgt_w
            if cur_h < tgt_h or cur_w < tgt_w:
                pad_h2 = tgt_h - cur_h
                pad_w2 = tgt_w - cur_w
                x = F.pad(x, (pad_w2 // 2, pad_w2 - pad_w2 // 2,
                              pad_h2 // 2, pad_h2 - pad_h2 // 2),
                          mode="reflect")
            x = adapt(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        x = self.head_dropout(x)

        if self.share_head:
            out = self.out_conv(x)  # (B, n_maps, H_pad, W_pad)
        else:
            feat = self.shared_proj(x)
            out = torch.cat([h(feat) for h in self.heads], dim=1)

        out = out[:, :, :H, :W]  # (B, n_maps, H, W)

        if self.out_patch is not None:
            P_out = self.out_patch
            cur_H, cur_W = out.shape[2], out.shape[3]
            h0 = (cur_H - P_out) // 2
            w0 = (cur_W - P_out) // 2
            out = out[:, :, h0:h0 + P_out, w0:w0 + P_out]

        # Output layout: (B, H, W, n_maps) for "patch" to match spectra-mode pattern
        if self.out_mode == "patch":
            return out.permute(0, 2, 3, 1).contiguous()  # (B, P, P, n_maps)
        elif self.out_mode == "flat":
            return out.reshape(B, -1)
        else:  # avg
            return out.mean(dim=(2, 3))  # (B, n_maps)

# ----------------- Spectra-mode helpers and models ----------------- #

def apply_derivatives_simple(X, der_mode="raw", ftir_wls=None, eps=1e-12):
    """
    Backward-difference derivatives with first-sample padding.
    Memory-optimized: only computes what is needed.
    """
    if der_mode == "raw":
        return X

    X = np.asarray(X, dtype=np.float32)
    orig_shape = X.shape

    if X.ndim == 2:
        X_flat = X
        restore = lambda Z: Z.astype(np.float32)
    elif X.ndim == 4:
        N, P1, P2, C = orig_shape
        X_flat = X.reshape(-1, C)
        restore = lambda Z: Z.reshape(N, P1, P2, -1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported X.ndim={X.ndim}")

    C = X_flat.shape[-1]
    if C < 2:
        raise ValueError("Need at least 2 spectral samples")

    if ftir_wls is None:
        dw = None
    else:
        w = np.asarray(ftir_wls, dtype=np.float32).ravel()
        if w.size != C:
            raise ValueError(f"ftir_wls length {w.size} != C {C}")
        dw = np.diff(w)
        dw = np.where(np.abs(dw) < eps, eps, dw)

    def backward_diff_pad_first(a_flat):
        d = np.empty_like(a_flat, dtype=np.float32)
        if dw is None:
            d[:, 1:] = a_flat[:, 1:] - a_flat[:, :-1]
        else:
            d[:, 1:] = (a_flat[:, 1:] - a_flat[:, :-1]) / dw[None, :]
        d[:, 0] = d[:, 1]
        return d

    need_d1 = der_mode in ("der1", "raw+der1", "raw+der1+der2", "der1+der2")
    need_d2 = der_mode in ("der2", "raw+der2", "raw+der1+der2", "der1+der2")

    if need_d1 or need_d2:
        d1 = backward_diff_pad_first(X_flat)
    else:
        d1 = None

    if need_d2:
        d2 = backward_diff_pad_first(d1)
    else:
        d2 = None

    if not need_d1 and d1 is not None:
        del d1
        d1 = None

    if der_mode == "der1":
        out = d1
    elif der_mode == "der2":
        out = d2
    elif der_mode == "raw+der1":
        out = np.concatenate([X_flat, d1], axis=-1)
        del d1
    elif der_mode == "raw+der2":
        out = np.concatenate([X_flat, d2], axis=-1)
        del d2
    elif der_mode == "raw+der1+der2":
        out = np.concatenate([X_flat, d1, d2], axis=-1)
        del d1, d2
    elif der_mode == "der1+der2":
        out = np.concatenate([d1, d2], axis=-1)
        del d1, d2
    else:
        raise ValueError(f"Unknown der_mode: {der_mode}")

    return restore(out)


# ----- Spectra models (copied from spectra script) -----

class SpectrumMLP(nn.Module):
    def __init__(self, c_in, c_out, hidden=(512, 256, 128, 256, 512), dropout=0.1):
        super().__init__()
        layers = []
        d = c_in
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))

            d = h
        layers.append(nn.Linear(d, c_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SpectralCNN1D(nn.Module):
    def __init__(self, c_in, c_out, hidden=128, n_layers=4, dropout=0.1):
        super().__init__()
        layers = []
        layers.append(nn.Conv1d(1, hidden, kernel_size=7, padding=3))
        layers.append(nn.BatchNorm1d(hidden))
        layers.append(nn.GELU())
        for _ in range(n_layers - 1):
            layers.append(nn.Conv1d(hidden, hidden, kernel_size=5, padding=2))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*layers)
        self.resample = nn.AdaptiveAvgPool1d(c_out)
        self.head = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden // 2, 1, kernel_size=1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.resample(x)
        x = self.head(x)
        return x.squeeze(1)


class SpectralUNet1D(nn.Module):
    def __init__(self, c_in, c_out, base_ch=64, depth=3, dropout=0.1):
        super().__init__()
        self.depth = depth
        self.in_proj = nn.Sequential(
            nn.Conv1d(1, base_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_ch),
            nn.GELU(),
        )
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = base_ch
        for _ in range(depth):
            self.encoders.append(self._block(ch, ch * 2, dropout))
            self.pools.append(nn.Conv1d(ch * 2, ch * 2, kernel_size=2, stride=2))
            ch = ch * 2
        self.bottleneck = self._block(ch, ch * 2, dropout)
        ch_bot = ch * 2
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = ch_bot
        for _ in range(depth):
            self.upconvs.append(nn.ConvTranspose1d(ch, ch // 2, kernel_size=2, stride=2))
            self.decoders.append(self._block(ch, ch // 2, dropout))
            ch = ch // 2
        self.spec_resample = nn.AdaptiveAvgPool1d(c_out)
        self.out_proj = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(ch, 1, kernel_size=1),
        )

    def _block(self, in_ch, out_ch, dropout):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.in_proj(x)
        L = x.shape[2]
        factor = 2 ** self.depth
        L_pad = math.ceil(L / factor) * factor
        if L_pad > L:
            x = F.pad(x, (0, L_pad - L))
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[2] != skip.shape[2]:
                x = F.pad(x, (0, skip.shape[2] - x.shape[2]))
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        x = x[:, :, :L]
        x = self.spec_resample(x)
        x = self.out_proj(x)
        return x.squeeze(1)


class UNet2D_FTIR(nn.Module):
    """2D UNet with spectral encoder/decoder for spectra-mode targets."""
    def __init__(self, in_ch, out_ch, base_ch=64, dropout=0.1,
                 out_mode="patch", out_patch=None, depth=3,
                 spec_enc_ch=64, spec_dec_mid=128):
        super().__init__()
        assert out_mode in ["avg", "flat", "patch"]
        self.out_mode = out_mode
        self.out_patch = out_patch
        self.depth = depth
        self.out_ch = out_ch

        self.spec_enc = nn.Sequential(
            nn.Conv2d(in_ch, spec_enc_ch * 2, kernel_size=1),
            nn.BatchNorm2d(spec_enc_ch * 2),
            nn.GELU(),
            nn.Conv2d(spec_enc_ch * 2, spec_enc_ch, kernel_size=1),
            nn.BatchNorm2d(spec_enc_ch),
            nn.GELU(),
        )
        self.enc1 = self._block(spec_enc_ch, base_ch, dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._block(base_ch, base_ch * 2, dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._block(base_ch * 2, base_ch * 4, dropout)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = self._block(base_ch * 4, base_ch * 8, dropout)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, 2)
        self.dec3 = self._block(base_ch * 8, base_ch * 4, dropout)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, 2)
        self.dec2 = self._block(base_ch * 4, base_ch * 2, dropout)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, 2)
        self.dec1 = self._block(base_ch * 2, base_ch, dropout)
        self.spec_dec = nn.Sequential(
            nn.Conv2d(base_ch, spec_dec_mid, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(spec_dec_mid, spec_dec_mid, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(spec_dec_mid, out_ch, kernel_size=1),
        )

    def _block(self, in_ch, out_ch, dropout):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.spec_enc(x)
        factor = 2 ** self.depth
        H_pad = math.ceil(H / factor) * factor
        W_pad = math.ceil(W / factor) * factor
        x = F.pad(x, (0, W_pad - W, 0, H_pad - H))
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.spec_dec(d1)
        out = out[:, :, :H, :W]
        if self.out_patch is not None and self.out_patch < H:
            P = self.out_patch
            h0 = (H - P) // 2
            w0 = (W - P) // 2
            out = out[:, :, h0:h0 + P, w0:w0 + P]
        if self.out_mode == "patch":
            return out.permute(0, 2, 3, 1).contiguous()  # (B, P, P, C_out)
        elif self.out_mode == "flat":
            return out.reshape(B, -1)
        else:
            return out.mean(dim=(2, 3))


class PLSRWrapperSpectra:
    """PLSR for vector targets (spectra-mode)."""
    def __init__(self, n_components=10):
        self.model = PLSRegression(n_components=n_components, scale=False)
        self.fitted = False
        self.y_shape_ = None

    def fit(self, X, y):
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            Y2d = y_arr.reshape(-1, 1)
            self.y_shape_ = ()
        else:
            self.y_shape_ = y_arr.shape[1:]
            Y2d = y_arr.reshape(y_arr.shape[0], -1)
        self.model.fit(X, Y2d)
        self.fitted = True

    def predict(self, X):
        Y2d = self.model.predict(X)
        if self.y_shape_ == () or self.y_shape_ is None:
            return Y2d.ravel()
        return Y2d.reshape((X.shape[0],) + self.y_shape_)


# ----- Spectra visualization helpers (copied from spectra script) -----

_fig_refs, _ax_refs, _im_refs, _slider_refs = [], [], [], []


def _limits_from_pair(d_true, d_pred, plot_range="minmax",
                      for_error=False, error_mode="signed"):
    if not for_error:
        if plot_range == "percentile":
            lo = min(np.nanpercentile(d_true, 2), np.nanpercentile(d_pred, 2))
            hi = max(np.nanpercentile(d_true, 98), np.nanpercentile(d_pred, 98))
        else:
            lo = np.nanmin([np.nanmin(d_true), np.nanmin(d_pred)])
            hi = np.nanmax([np.nanmax(d_true), np.nanmax(d_pred)])
        return lo, hi
    err = d_pred - d_true
    if error_mode == "abs":
        err = np.abs(err)
        if plot_range == "percentile":
            return 0.0, np.nanpercentile(err, 98)
        return 0.0, float(np.nanmax(err))
    if plot_range == "percentile":
        m = max(abs(np.nanpercentile(err, 2)), abs(np.nanpercentile(err, 98)))
    else:
        m = float(np.nanmax(np.abs(err)))
    return -m, m


def _connect_slider(sl, im_gt, im_pr, im_er, ax_gt, ax_pr, ax_er,
                    waves, d_true, d_pred, error_mode="signed"):
    err_cube = d_pred - d_true
    if error_mode == "abs":
        err_cube = np.abs(err_cube)

    def _upd(val):
        k = int(sl.val)
        im_gt.set_data(d_true[:, :, k])
        ax_gt.set_title(f"Ground truth\n{waves[k]:.0f}")
        im_pr.set_data(d_pred[:, :, k])
        ax_pr.set_title(f"Prediction\n{waves[k]:.0f}")
        im_er.set_data(err_cube[:, :, k])
        ax_er.set_title(("Error (Pred − GT)" if error_mode == "signed" else "|Error|")
                        + f"\n{waves[k]:.0f}")
        im_gt.figure.canvas.draw_idle()

    sl.on_changed(_upd)


def show_slider(d_true, d_pred, wavelengths, test_cols,
                cmap="viridis", plot_range="percentile",
                error_mode="signed",
                err_cmap_signed="coolwarm", err_cmap_abs="magma"):
    global _fig_refs, _ax_refs, _im_refs, _slider_refs
    if matplotlib.get_backend().lower() == "agg":
        return

    if d_true.ndim != 3 or d_pred.ndim != 3:
        return
    d_true = d_true[:, test_cols[0]:test_cols[1], :]
    d_pred = d_pred[:, test_cols[0]:test_cols[1], :]
    if d_pred.shape != d_true.shape:
        return

    _, _, C = d_true.shape

    vmin, vmax = _limits_from_pair(d_true, d_pred, plot_range, for_error=False)
    emin, emax = _limits_from_pair(d_true, d_pred, plot_range,
                                   for_error=True, error_mode=error_mode)
    err_cmap = err_cmap_signed if error_mode == "signed" else err_cmap_abs

    init = 0
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    plt.subplots_adjust(bottom=0.20)
    im1 = ax1.imshow(d_true[:, :, init], cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_title(f"Ground truth\n{wavelengths[init]:.0f}"); ax1.axis("off")
    im2 = ax2.imshow(d_pred[:, :, init], cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_title(f"Prediction\n{wavelengths[init]:.0f}"); ax2.axis("off")
    init_err = d_pred[:, :, init] - d_true[:, :, init]
    if error_mode == "abs":
        init_err = np.abs(init_err)
    im3 = ax3.imshow(init_err, cmap=err_cmap, vmin=emin, vmax=emax)
    ax3.axis("off")
    fig.colorbar(im1, ax=[ax1, ax2], fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax_sl = plt.axes([0.20, 0.05, 0.60, 0.04])
    sl = _MplSlider(ax_sl, "Band", 0, C - 1, valinit=init, valstep=1)
    _connect_slider(sl, im1, im2, im3, ax1, ax2, ax3,
                    wavelengths, d_true, d_pred, error_mode=error_mode)
    _fig_refs.append(fig); _ax_refs += [ax1, ax2, ax3]
    _im_refs += [im1, im2, im3]; _slider_refs.append(sl)


def build_pred_cube_from_outputs(ftir_cube, coords_test, y_pred_test,
                                 C_ftir, out_patch):
    ftir_cube = np.asarray(ftir_cube, dtype=np.float32)
    Hc, Wc, _ = ftir_cube.shape
    y_pred_test = np.asarray(y_pred_test, dtype=np.float32)
    pred_sum = np.zeros_like(ftir_cube, dtype=np.float32)
    weight = np.zeros((Hc, Wc), dtype=np.float32)

    for idx in range(y_pred_test.shape[0]):
        r, c = coords_test[idx]
        if not (0 <= r < Hc and 0 <= c < Wc):
            continue
        pred = y_pred_test[idx]
        if pred.ndim == 1:
            tot = pred.shape[0]
            if tot == C_ftir:
                patch = pred.reshape(1, 1, C_ftir); P = 1
            elif tot % C_ftir == 0:
                P2 = tot // C_ftir
                P = int(round(np.sqrt(P2)))
                if P * P != P2:
                    continue
                patch = pred.reshape(P, P, C_ftir)
            else:
                continue
        elif pred.ndim == 3:
            P1, P2, C_out = pred.shape
            if P1 != P2 or C_out < C_ftir:
                continue
            P = P1
            patch = pred[..., :C_ftir]
        else:
            continue
        if out_patch is not None and P != out_patch:
            continue
        rad = P // 2
        r0, r1 = r - rad, r - rad + P
        c0, c1 = c - rad, c - rad + P
        if r0 < 0 or c0 < 0 or r1 > Hc or c1 > Wc:
            continue
        pred_sum[r0:r1, c0:c1, :] += patch
        weight[r0:r1, c0:c1] += 1.0

    w3 = np.maximum(weight[..., None], 1e-6)
    pred_cube = pred_sum / w3
    pred_cube[weight == 0] = np.nan
    if np.all(np.isnan(pred_cube)):
        return None
    return pred_cube


def plot_spectra_at_points(points, ftir_cube, coords_test, y_pred_test,
                           wavelengths, save_path=None):
    C = ftir_cube.shape[2]
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure()
    for k, p in enumerate(points):
        if isinstance(p, (int, np.integer)):
            idx = p
            if idx < 0 or idx >= len(coords_test):
                continue
            r, c = coords_test[idx]
        else:
            r, c = p
            try:
                idx = coords_test.index((r, c))
            except ValueError:
                continue
        gt = ftir_cube[r, c, :].astype(np.float32)
        pred_full = np.asarray(y_pred_test[idx], dtype=np.float32)
        if pred_full.ndim == 1:
            pred = pred_full
        elif pred_full.ndim == 3:
            P1, P2, Cp = pred_full.shape
            if Cp < C:
                continue
            pred = pred_full[P1 // 2, P2 // 2, :C]
        else:
            continue
        col = color_cycle[k % len(color_cycle)]
        plt.plot(wavelengths, gt, color=col, lw=1.5)
        plt.plot(wavelengths, pred, color=col, lw=1.5, linestyle="--")
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Absorbance (a.u.)")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ----------------- Run tag ----------------- #
REGISTRY_BASENAME = "run_registry.xlsx"

def _flatten_for_excel(d, parent_key="", sep="."):
    """
    Flatten nested dicts/lists into single-level keys for one Excel row.
    Examples:
        {"a": {"b": 1}}   -> {"a.b": 1}
        {"a": [1, 2]}     -> {"a": "1, 2"}
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(_flatten_for_excel(v, new_key, sep=sep))
        elif isinstance(v, (list, tuple)):
            items[new_key] = ", ".join(str(x) for x in v)
        else:
            items[new_key] = v
    return items

def _args_to_excel_row(args):
    """Convert argparse Namespace into a flat dict with 'arg.<name>' keys."""
    out = {}
    for k, v in vars(args).items():
        if isinstance(v, (list, tuple)):
            out[f"arg.{k}"] = ", ".join(str(x) for x in v)
        else:
            out[f"arg.{k}"] = v
    return out


def append_run_to_excel(registry_path, row_dict, sheet_name="runs",
                        key_cols=("short_tag", "fold_tag", "results_dir")):
    """
    Append a row to the registry Excel file.
    Resilient to corrupted .xlsx files (e.g. from OneDrive sync issues):
      - On read failure, the corrupted file is BACKED UP (not silently overwritten).
      - Each row is also appended to a CSV mirror that never gets rewritten,
        so no run is ever lost.
    """
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)

    # ---- Always append to a CSV mirror (safe, never rewritten) ----
    csv_path = registry_path.replace(".xlsx", ".csv")
    new_df = pd.DataFrame([row_dict])
    try:
        write_header = not os.path.exists(csv_path)
        new_df.to_csv(csv_path, mode="a", header=write_header, index=False)
    except Exception as e:
        print(f"[registry] CSV mirror write failed: {e}")

    # ---- Try to maintain the .xlsx file too (best effort) ----
    if os.path.exists(registry_path):
        try:
            old_df = pd.read_excel(registry_path, sheet_name=sheet_name)
        except Exception as e:
            # Corrupt file (often from OneDrive). Back it up rather than
            # silently overwrite, so we don't destroy old rows.
            ts = time.strftime("%Y%m%d_%H%M%S")
            backup_path = f"{registry_path}.corrupt_{ts}.bak"
            try:
                os.replace(registry_path, backup_path)
                print(f"[registry] Could not read {registry_path}: {e}")
                print(f"[registry] Corrupted file backed up to: {backup_path}")
            except Exception as e2:
                print(f"[registry] Could not back up corrupt file: {e2}")

            # Recover from CSV mirror if available
            if os.path.exists(csv_path):
                try:
                    old_df = pd.read_csv(csv_path)
                    print(f"[registry] Recovered {len(old_df)} rows from CSV mirror.")
                except Exception:
                    old_df = pd.DataFrame()
            else:
                old_df = pd.DataFrame()
    else:
        # First run: try to seed from existing CSV mirror if it exists
        if os.path.exists(csv_path):
            try:
                old_df = pd.read_csv(csv_path)
            except Exception:
                old_df = pd.DataFrame()
        else:
            old_df = pd.DataFrame()

    # Drop the prior row matching this same key, then append the new one
    if not old_df.empty and all(c in old_df.columns for c in key_cols):
        mask = pd.Series([True] * len(old_df))
        for c in key_cols:
            mask &= old_df[c] == row_dict.get(c)
        old_df = old_df[~mask]

    combined = pd.concat([old_df, new_df], ignore_index=True, sort=False)

    priority = ["timestamp", "short_tag", "fold_tag", "results_dir", "run_tag"]
    cols = [c for c in priority if c in combined.columns] + \
           [c for c in combined.columns if c not in priority]
    combined = combined[cols]

    if _HAS_FCNTL:
        lock_path = registry_path + ".lock"
        try:
            with open(lock_path, "w") as lock_fh:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
                # ... re-read and write inside the lock ...
                with pd.ExcelWriter(registry_path, engine="openpyxl",
                                    mode="w") as writer:
                    combined.to_excel(writer, sheet_name=sheet_name,
                                      index=False)
        except Exception as e:
            print(f"[registry] Excel write failed: {e}")
            print(f"[registry] CSV mirror at {csv_path} "
                  f"still contains all data.")
    else:
        # Windows / no fcntl: write without locking.
        # Safe for single-process use; if you run multiple writers
        # concurrently on Windows, results may be lost (but CSV mirror
        # is still appended).
        try:
            with pd.ExcelWriter(registry_path, engine="openpyxl",
                                mode="w") as writer:
                combined.to_excel(writer, sheet_name=sheet_name,
                                  index=False)
        except Exception as e:
            print(f"[registry] Excel write failed: {e}")
            print(f"[registry] CSV mirror at {csv_path} "
                  f"still contains all data.")

def make_run_tag(args):
    if (getattr(args, "model", "") == "multi_unet"
            and getattr(args, "map_fields", None)):
        map_label = "multi-" + "-".join(args.map_fields)
    else:
        map_label = args.map_field
    parts = [
        f"tgt-{args.target_mode}",
        f"{map_label}",
        f"{args.model}",
        f"loss-{args.loss}",
        f"test-{args.test_prefix}",
        f"val-{args.val_prefix}",
        f"inPatch-{args.patch}",
        f"inAvg-{args.avg_mode}",
        f"outPatch-{args.out_patch}",
        f"outAvg-{args.out_avg_mode}",
        f"der-{args.der_mode}",
        f"dr-{args.dr_mode}{args.dr_components if args.dr_mode != 'none' else ''}",
        f"base-{args.baseline_mode}",
        f"sched-{args.scheduler}",
        f"xScaler-{args.x_scaler}",
        f"yScaler-{args.y_scaler}",
        f"smoothX-{int(args.smooth_x)}",
    ]
    # SG is now used whenever smoothing OR derivatives are active
    _sg_used = args.smooth_x or args.der_mode != "raw" or \
            (args.target_mode == "spectra" and getattr(args, "smooth_y", False))
    if _sg_used:
        parts.append(f"sgW-{args.sg_window}")
        parts.append(f"sgP-{args.sg_poly}")
    if args.spectrum_norm != "none":
        parts.append(f"specNorm-{args.spectrum_norm}")
    if args.model == "mlp":
        parts.append("mlpH-" + "x".join(str(h) for h in args.mlp_hidden))
    elif args.model == "cnn1d":
        parts.append(f"c1dH-{args.cnn1d_hidden}_L-{args.cnn1d_layers}")
    elif args.model == "cnn":
        parts.append("c2dH-" + "x".join(str(h) for h in args.cnn_hidden))
    elif args.model == "plsr":
        parts.append(f"plsrC-{args.plsr_components}")
    elif args.model in ("unet", "unet_simple", "multi_unet"):
        parts.append(f"unetBase-{args.unet_base_ch}")
        parts.append(f"unetD-{args.unet_depth}")
        if args.target_mode == "spectra" and args.model == "unet":
            parts.append("unetHead-spectra")
    if any([args.grad_weight, args.lap_weight, args.var_weight, args.fft_weight]):
        # Compact tag: only show non-zero components.
        sharp_parts = []
        if args.grad_weight: sharp_parts.append(f"g{args.grad_weight}")
        if args.lap_weight:  sharp_parts.append(f"l{args.lap_weight}")
        if args.var_weight:  sharp_parts.append(f"v{args.var_weight}")
        if args.fft_weight:  sharp_parts.append(f"f{args.fft_weight}")
        parts.append("sharp-" + "_".join(sharp_parts))
    if args.target_mode == "spectra":
        parts.append(f"outDer-{args.out_der_mode}")
        parts.append(f"smoothY-{int(args.smooth_y)}")
    if args.spatial_loss_summary != "mean":
        parts.append(f"sumLoss-{args.spatial_loss_summary}")
        if args.spatial_loss_bands:
            parts.append("bands-" + "-".join(str(b) for b in args.spatial_loss_bands))
    if args.model == "pretrained_unet":
        parts.append(f"enc-{args.pretrained_encoder}")
        parts.append(f"freeze-{args.freeze_encoder_epochs}")
        parts.append(f"ftLR-{args.finetune_lr_mult}")

    if args.add_depth_channel: parts.append("depthCh")
    if args.add_pos_channels:  parts.append("posCh")
    if args.add_global_channel:
        mode = getattr(args, "global_mode", "neighborhood")
        if mode == "full_image_resize":
            parts.append(f"globCh-fullResize_pc{args.global_channel_pcs}")
        else:
            parts.append(f"globCh-nbhd-s{args.global_lowres_scale}"
                         f"_pc{args.global_channel_pcs}")
    if getattr(args, "add_mask_channel", False):
        parts.append("maskCh")
    if getattr(args, "nonmask_fill", "keep_zero") != "keep_zero":
        parts.append(f"fill-{args.nonmask_fill}")
    if getattr(args, "exclude_wl_ranges", None):
        excl_str = "-".join(
            f"{args.exclude_wl_ranges[i]:.0f}to{args.exclude_wl_ranges[i+1]:.0f}"
            for i in range(0, len(args.exclude_wl_ranges), 2)
        )
        parts.append(f"exclWL-{excl_str}")


    return "_".join(parts)

def make_short_tag(args, run_tag, length=8):
    if (getattr(args, "model", "") == "multi_unet"
            and getattr(args, "map_fields", None)):
        map_label = "multi-" + "-".join(args.map_fields)
    else:
        map_label = args.map_field
    h = hashlib.md5(run_tag.encode("utf-8")).hexdigest()[:length]
    short = (
        f"tgt-{args.target_mode}_"
        f"{args.model}_{map_label}_test-{args.test_prefix}_{h}"
    )
    return short


def unique_dir(path):
    """
    Return `path` if it does not exist, otherwise append _v2, _v3, ...
    until a non-existing path is found. Does NOT create the directory.
    """
    if not os.path.exists(path):
        return path
    i = 2
    while True:
        candidate = f"{path}_v{i}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


# ----------------- Metrics ----------------- #

def local_std_ratio(gt, pred, mask, k=5):
    """
    Ratio of mean local std (k x k) of prediction vs GT, over the mask.
    Supports 2D maps (H, W) and 3D spectral cubes (H, W, C); in the 3D
    case the spectral axis is collapsed with nanmean before computing
    local std, matching the 2D summary used for the BO blur penalty
    elsewhere in this codebase.
    """
    gt = np.asarray(gt, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)

    # Collapse spectral axis if present, so we always compare 2D summaries.
    if gt.ndim == 3:
        with np.errstate(invalid="ignore"):
            gt = np.nanmean(gt, axis=-1)
    if pred.ndim == 3:
        with np.errstate(invalid="ignore"):
            pred = np.nanmean(pred, axis=-1)

    if gt.shape != pred.shape or gt.shape != mask.shape:
        return float("nan")

    def _lstd(a):
        a = np.where(np.isfinite(a), a, 0.0)
        mu = uniform_filter(a, size=k)
        mu2 = uniform_filter(a * a, size=k)
        return np.sqrt(np.clip(mu2 - mu ** 2, 0.0, None))

    s_g = _lstd(gt)
    s_p = _lstd(pred)
    sel = mask & np.isfinite(s_g) & np.isfinite(s_p)
    if not sel.any():
        return float("nan")
    return float(np.mean(s_p[sel]) / (np.mean(s_g[sel]) + 1e-12))


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Drop non-finite pairs. Use a single fused isfinite check that
    # short-circuits memory allocation when the inputs are already clean
    # (the common case after sanitize_inplace).
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        if not finite.any():
            return np.nan, np.nan, np.nan, np.nan
        y_true = y_true[finite]
        y_pred = y_pred[finite]


    mae = float(np.mean(np.abs(y_pred - y_true)))
    mse = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))
    if y_true.size > 1 and np.std(y_true) > 0:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = np.nan
    return mae, mse, rmse, r2

class MaskedRegressionLoss(nn.Module):
    def __init__(self, loss_type="mse", smooth_weight=0.0, max_shift=0,
                 huber_delta=1.0, y_bins=None, y_weights=None,
                 grad_weight=0.0, lap_weight=0.0,
                 var_weight=0.0, var_kernel=5,
                 fft_weight=0.0,
                 spatial_loss_summary="mean",
                 spatial_loss_bands=None,
                 per_channel_spatial=False,
                 spectral_smooth_weight=0.0):
        super().__init__()
        self.loss_type = loss_type
        self.smooth_weight = float(smooth_weight)
        self.max_shift = int(max_shift)
        self.huber_delta = float(huber_delta)
        self.grad_weight = float(grad_weight)
        self.lap_weight  = float(lap_weight)
        self.var_weight  = float(var_weight)
        self.var_kernel  = int(var_kernel)
        self.fft_weight  = float(fft_weight)
        self.per_channel_spatial = bool(per_channel_spatial)
        self.spectral_smooth_weight = float(spectral_smooth_weight)

        # How to collapse (B,P,P,C_out) → (B,P,P) for max_shift / edge losses
        assert spatial_loss_summary in ("mean", "bands")
        self.spatial_loss_summary = spatial_loss_summary
        if spatial_loss_bands is not None and len(spatial_loss_bands) > 0:
            bands = torch.as_tensor(list(spatial_loss_bands), dtype=torch.long)
        else:
            bands = None

        if bands is not None:
            self.register_buffer("spatial_loss_bands", bands)
        else:
            # plain attribute (not a buffer) so it doesn't try to move with .to()
            self.spatial_loss_bands = None

        if y_bins is not None and y_weights is not None and len(y_bins) > 0:
            self.register_buffer("y_bins", torch.as_tensor(y_bins, dtype=torch.float32))
            self.register_buffer("y_weights", torch.as_tensor(y_weights, dtype=torch.float32))
            self._has_weights = True
        else:
            self.y_bins = None
            self.y_weights = None
            self._has_weights = False

        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3).clone()
        lap = torch.tensor([[0.,  1., 0.],
                            [1., -4., 1.],
                            [0.,  1., 0.]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("lap_k",   lap)

    # ---------- helpers ----------
    def _pixel_loss(self, pred, target):
        if self.loss_type == "mse":
            return (pred - target) ** 2
        elif self.loss_type == "l1":
            return (pred - target).abs()
        elif self.loss_type == "huber":
            return F.smooth_l1_loss(pred, target, reduction="none",
                                    beta=self.huber_delta)
        elif self.loss_type == "wmse":
            return (pred - target) ** 2
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def _weights_for(self, target):
        if not getattr(self, "_has_weights", False):
            return torch.ones_like(target)
        if (self.y_bins is None or self.y_weights is None
                or self.y_weights.numel() == 0
                or self.y_bins.numel() == 0):
            return torch.ones_like(target)
        # In spectra mode, target has a spectral axis (ndim == 4). A single
        # global y-histogram conflates absorbance values across wavenumbers
        # (e.g. an Amide I peak vs. a flat baseline). Fall back to uniform
        # weights in that case; per-channel weighting would require a
        # per-channel y_bins/y_weights table, which the current pipeline
        # does not build.
        if target.ndim == 4:
            return torch.ones_like(target)
        safe_target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        idx = torch.bucketize(safe_target, self.y_bins)
        idx = idx.clamp(0, self.y_weights.numel() - 1)
        return self.y_weights[idx]



    def _masked_mean(self, per_pix, mask, weights=None):
        if mask is None:
            if weights is not None:
                return (per_pix * weights).sum() / (weights.sum() + 1e-8)
            return per_pix.mean()
        sel = per_pix[mask]
        if sel.numel() == 0:
            return None
        if weights is not None:
            w = weights[mask]
            return (sel * w).sum() / (w.sum() + 1e-8)
        return sel.mean()


    # ---- per-channel reshape for spectra-mode spatial losses ----
    def _per_channel_2d(self, t):
        """
        Reshape (B, P, P, C_out) -> (B*C_out, P, P) so existing 2D
        Sobel/Laplacian/var/FFT kernels apply per spectral channel.
        For 3D (map mode), return unchanged.
        """
        if t.ndim == 3:
            return t, None
        if t.ndim == 4:
            B, H, W, C = t.shape
            return t.permute(0, 3, 1, 2).reshape(B * C, H, W), (B, C)
        raise ValueError(f"_per_channel_2d: unsupported ndim={t.ndim}")

    # ---- spectral-axis smoothness penalty (spectra mode only) ----
    def _spectral_smooth_loss(self, pred, mask):
        """
        L2 penalty on first-order differences along the spectral axis.
        pred: (B, P, P, C_out) or (B, C_out).
        mask: spatial mask broadcastable over channel; ignored if None.
        """
        if pred.ndim == 4:
            ds = (pred[..., 1:] - pred[..., :-1]) ** 2  # (B, P, P, C_out-1)
            if mask is not None and mask.ndim == 3:
                m = mask.unsqueeze(-1).expand(ds.shape[0], ds.shape[1],
                                              ds.shape[2], ds.shape[3])
                if m.dtype != torch.bool:
                    m = m > 0
                sel = ds[m]
                return sel.mean() if sel.numel() > 0 else pred.sum() * 0.0
            return ds.mean()
        elif pred.ndim == 2:
            ds = (pred[:, 1:] - pred[:, :-1]) ** 2
            return ds.mean()
        else:
            return pred.sum() * 0.0

    # ---- collapse 4D -> 2D summary used by max_shift / edge losses ----
    def _summarize2d(self, t):

        """
        Convert (B,P,P) -> (B,P,P), or (B,P,P,C_out) -> (B,P,P).
        Used only for max_shift search and edge-enhanced terms.
        """
        if t.ndim == 3:
            return t
        if t.ndim == 4:
            if self.spatial_loss_summary == "mean":
                return t.mean(dim=-1)
            elif self.spatial_loss_summary == "bands":
                if self.spatial_loss_bands is None:
                    return t.mean(dim=-1)
                idx = self.spatial_loss_bands
                idx = idx.clamp(0, t.shape[-1] - 1)
                return t.index_select(-1, idx).mean(dim=-1)
        raise ValueError(f"_summarize2d expects 3D or 4D, got ndim={t.ndim}")

    # ---- mask shape normalizer to (B, P, P) ----
    def _mask2d(self, mask, target_2d_shape):
        """
        Make sure the mask is (B, P, P) regardless of how it was passed in.
        target_2d_shape is the (B, P, P) shape we want.
        """
        if mask is None:
            return None
        if mask.ndim == 3 and mask.shape == target_2d_shape:
            return mask
        if mask.ndim == 4:
            # collapse a per-channel broadcast back to spatial
            return mask.any(dim=-1) if mask.dtype == torch.bool else (mask > 0).any(dim=-1)
        return mask  # caller will broadcast if needed

    # ---------- spatial terms (operate on 2D summaries) ----------
    def _erode_mask(self, mask2d, radius=1):
        if mask2d is None:
            return None
        if radius <= 0:
            return mask2d
        m = mask2d.clone()
        for _ in range(radius):
            m = (m
                & torch.roll(m,  1, dims=1) & torch.roll(m, -1, dims=1)
                & torch.roll(m,  1, dims=2) & torch.roll(m, -1, dims=2))
            # Zero out wrap-around at borders
            m[:,  0, :] = False; m[:, -1, :] = False
            m[:, :,  0] = False; m[:, :, -1] = False
        return m


    def _grad_loss(self, pred2d, target2d, mask2d):
        p = pred2d.unsqueeze(1)
        t = target2d.unsqueeze(1)
        gx_p = F.conv2d(p, self.sobel_x, padding=1)
        gy_p = F.conv2d(p, self.sobel_y, padding=1)
        gx_t = F.conv2d(t, self.sobel_x, padding=1)
        gy_t = F.conv2d(t, self.sobel_y, padding=1)
        diff = ((gx_p - gx_t).abs() + (gy_p - gy_t).abs()).squeeze(1)
        m_e = self._erode_mask(mask2d, radius=1) if mask2d is not None and mask2d.ndim == 3 else mask2d
        v = self._masked_mean(diff, m_e)
        return v if v is not None else pred2d.sum() * 0.0

    def _lap_loss(self, pred2d, target2d, mask2d):
        p = pred2d.unsqueeze(1)
        t = target2d.unsqueeze(1)
        lp = F.conv2d(p, self.lap_k, padding=1).squeeze(1)
        lt = F.conv2d(t, self.lap_k, padding=1).squeeze(1)
        diff = (lp - lt).abs()
        m_e = self._erode_mask(mask2d, radius=1) if mask2d is not None and mask2d.ndim == 3 else mask2d
        v = self._masked_mean(diff, m_e)
        return v if v is not None else pred2d.sum() * 0.0

    def _local_var_loss(self, pred2d, target2d, mask2d):
        k = self.var_kernel
        pad = k // 2
        kernel = torch.ones(1, 1, k, k, device=pred2d.device) / (k * k)
        p = pred2d.unsqueeze(1)
        t = target2d.unsqueeze(1)
        mu_p  = F.conv2d(p,    kernel, padding=pad)
        mu_t  = F.conv2d(t,    kernel, padding=pad)
        mu_p2 = F.conv2d(p*p,  kernel, padding=pad)
        mu_t2 = F.conv2d(t*t,  kernel, padding=pad)
        eps = 1e-6
        var_p = (mu_p2 - mu_p ** 2).clamp_min(eps)
        var_t = (mu_t2 - mu_t ** 2).clamp_min(eps)
        diff = (var_p.sqrt() - var_t.sqrt()).abs().squeeze(1)
        # Erode by the variance-kernel radius so windows don't include fold pixels
        m_e = self._erode_mask(mask2d, radius=pad) if mask2d is not None and mask2d.ndim == 3 else mask2d
        v = self._masked_mean(diff, m_e)
        return v if v is not None else pred2d.sum() * 0.0

    def _fft_loss(self, pred2d, target2d, mask2d):
        if mask2d is None:
            m = torch.ones_like(pred2d)
            p_in = pred2d
            t_in = target2d
        else:
            mask_bool = mask2d if mask2d.dtype == torch.bool else (mask2d > 0)
            m = mask_bool.float()
            denom = m.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
            p_mu = (pred2d * m).sum(dim=(1, 2), keepdim=True) / denom
            t_mu = (target2d * m).sum(dim=(1, 2), keepdim=True) / denom
            p_in = torch.where(mask_bool, pred2d, p_mu.expand_as(pred2d))
            t_in = torch.where(mask_bool, target2d, t_mu.expand_as(target2d))


        p = torch.nan_to_num(p_in, nan=0.0, posinf=0.0, neginf=0.0)
        t = torch.nan_to_num(t_in, nan=0.0, posinf=0.0, neginf=0.0)
        Fp = torch.fft.rfft2(p, norm="ortho")
        Ft = torch.fft.rfft2(t, norm="ortho")
        return (Fp.abs() - Ft.abs()).abs().mean()

    # ---------- shift helpers ----------
    @staticmethod
    def _crop_center(t, k):
        """Crop k pixels from each spatial border. Works for 3D and 4D."""
        if t.ndim == 3:
            return t[:, k:-k, k:-k]
        return t[:, k:-k, k:-k, ...]

    @staticmethod
    def _crop_shift(t, k, dr, dc):
        """Crop a window of size H_c x W_c at offset (k+dr, k+dc)."""
        if t.ndim == 3:
            H_c = t.shape[1] - 2 * k
            W_c = t.shape[2] - 2 * k
            r0, c0 = k + dr, k + dc
            return t[:, r0:r0 + H_c, c0:c0 + W_c]
        if t.ndim == 4:
            H_c = t.shape[1] - 2 * k
            W_c = t.shape[2] - 2 * k
            r0, c0 = k + dr, k + dc
            return t[:, r0:r0 + H_c, c0:c0 + W_c, ...]
        raise ValueError(f"_crop_shift unsupported ndim={t.ndim}")

    # ---------- forward ----------
    def forward(self, pred, target, mask):
        """
        pred, target:
            (B, H, W)            map mode
            (B, H, W, C_out)     spectra mode
        mask:
            (B, H, W)            spatial mask (preferred; we use this 2D form)
            (B, H, W, C_out)     also accepted; collapsed via .any(-1)

        Strategy:
          1) Build 2D summaries (mean or selected bands) of pred/target for
             max_shift search and edge-enhanced terms.
          2) If max_shift > 0, find the best (dr, dc) that minimizes the
             pixel-wise data fidelity on the 2D summary; apply that one
             shift to the full pred/target/mask for the final pixel loss
             (which is per-element, full 4D in spectra mode).
          3) Add edge-enhanced terms on the (shifted) 2D summary. They use
             a per-channel-aggregate signal, which is much cheaper and
             usually more stable than per-channel sharpness losses.
        """
        # 2D summary tensors (used by max_shift & edge losses)
        pred2d   = self._summarize2d(pred)
        target2d = self._summarize2d(target)
        # 2D mask
        if mask is not None and mask.ndim == 4:
            mask2d_full = mask.any(dim=-1) if mask.dtype == torch.bool \
                else (mask > 0).any(dim=-1)
        else:
            mask2d_full = mask

        # ---------- Best-shift search on 2D summary ----------
        best_shift = (0, 0)
        if self.max_shift > 0:
            k = self.max_shift
            # Crop the prediction summary once. Detach so we don't keep
            # autograd graph for the candidate evaluations: the data term
            # and the edge terms below carry the real gradient at the
            # *chosen* shift only.
            p2d_c = self._crop_center(pred2d, k).detach()
            best_v = None
            with torch.no_grad():
                for dr in range(-k, k + 1):
                    for dc in range(-k, k + 1):
                        t2d_c = self._crop_shift(target2d, k, dr, dc)
                        m2d_c = self._crop_shift(mask2d_full, k, dr, dc) \
                            if mask2d_full is not None else None
                        if m2d_c is not None and not m2d_c.any():
                            continue
                        per_pix = self._pixel_loss(p2d_c, t2d_c)
                        w = self._weights_for(t2d_c) if self.loss_type == "wmse" else None
                        v = self._masked_mean(per_pix, m2d_c, w)
                        if v is None:
                            continue
                        if best_v is None or float(v) < float(best_v):
                            best_v = v
                            best_shift = (dr, dc)

            # Apply the chosen shift consistently to ALL terms
            dr, dc = best_shift
            pred_c    = self._crop_center(pred, k)
            target_c  = self._crop_shift(target, k, dr, dc)
            mask_c    = self._crop_shift(mask, k, dr, dc) if mask is not None else None
            pred2d_c   = self._crop_center(pred2d, k)
            target2d_c = self._crop_shift(target2d, k, dr, dc)
            mask2d_c   = self._crop_shift(mask2d_full, k, dr, dc) \
                if mask2d_full is not None else None
        else:
            pred_c    = pred
            target_c  = target
            mask_c    = mask
            pred2d_c   = pred2d
            target2d_c = target2d
            mask2d_c   = mask2d_full

        # ---------- Pixel-wise data fidelity on the FULL tensors ----------
        per_pix = self._pixel_loss(pred_c, target_c)
        if mask_c is not None and mask_c.ndim != per_pix.ndim:
            # broadcast a 3D mask over channel for 4D pred
            if per_pix.ndim == 4 and mask_c.ndim == 3:
                mask_c_b = mask_c.unsqueeze(-1).expand_as(per_pix)
            else:
                mask_c_b = mask_c
        else:
            mask_c_b = mask_c

        w = (self._weights_for(target_c)
            if (self.loss_type == "wmse" and target_c.ndim < 4)
            else None)

        data_term = self._masked_mean(per_pix, mask_c_b, w) \
            if mask_c_b is not None else per_pix.mean()
        if data_term is None:
            return (pred * 0.0).sum()

        total = data_term

        # ---------- Smoothness on the 2D summary (no target) ----------
        if self.smooth_weight > 0:
            dy = (pred2d_c[:, 1:, :] - pred2d_c[:, :-1, :]) ** 2
            dx = (pred2d_c[:, :, 1:] - pred2d_c[:, :, :-1]) ** 2
            total = total + self.smooth_weight * (dy.mean() + dx.mean())

        # ---------- Edge-enhanced terms ----------
        if self.per_channel_spatial and pred_c.ndim == 4:
            # Per-channel spatial sharpness: apply 2D edge losses to each
            # wavenumber independently and average.
            p_pc, shape_info = self._per_channel_2d(pred_c)
            t_pc, _          = self._per_channel_2d(target_c)
            if mask_c is not None and mask_c.ndim == 3:
                B, H, W, C = pred_c.shape
                m_pc = mask_c.unsqueeze(-1).expand(B, H, W, C) \
                             .permute(0, 3, 1, 2).reshape(B * C, H, W)
            else:
                m_pc = None

            if self.grad_weight > 0:
                total = total + self.grad_weight * self._grad_loss(p_pc, t_pc, m_pc)
            if self.lap_weight > 0:
                total = total + self.lap_weight  * self._lap_loss(p_pc, t_pc, m_pc)
            if self.var_weight > 0:
                total = total + self.var_weight  * self._local_var_loss(p_pc, t_pc, m_pc)
            if self.fft_weight > 0:
                total = total + self.fft_weight  * self._fft_loss(p_pc, t_pc, m_pc)
        else:
            # Original behaviour: operate on a spectral summary (mean or bands).
            if self.grad_weight > 0:
                total = total + self.grad_weight * self._grad_loss(pred2d_c, target2d_c, mask2d_c)
            if self.lap_weight > 0:
                total = total + self.lap_weight  * self._lap_loss(pred2d_c, target2d_c, mask2d_c)
            if self.var_weight > 0:
                total = total + self.var_weight  * self._local_var_loss(pred2d_c, target2d_c, mask2d_c)
            if self.fft_weight > 0:
                total = total + self.fft_weight  * self._fft_loss(pred2d_c, target2d_c, mask2d_c)

        # ---------- Spectral-axis smoothness (spectra mode only) ----------
        if self.spectral_smooth_weight > 0 and pred_c.ndim >= 2:
            total = total + self.spectral_smooth_weight * \
                    self._spectral_smooth_loss(pred_c, mask_c)

        return total

# ----------------- Training loop ----------------- #

def train_torch_model_gan(
    model, train_loader, val_loader, device,
    epochs=20, lr=1e-3,
    scheduler_type="none", scheduler_kwargs=None,
    grad_clip=1.0, patience_early_stop=15,
    max_shift=0, huber_delta=1.0,
    y_bins=None, y_weights=None,
    grad_weight=0.0, lap_weight=0.0,
    var_weight=0.0, var_kernel=5, fft_weight=0.0,
    spatial_loss_summary="mean", spatial_loss_bands=None,
    y_scaler=None, y_scaler_kind="none",
    per_channel_spatial=False,
    spectral_smooth_weight=0.0,
    # GAN-specific
    gan_mode="pix2pix",
    gan_lambda_l1=100.0,
    gan_lambda_adv=1.0,
    gan_lambda_cycle=10.0,
    gan_lambda_identity=0.0,
    gan_d_base_ch=64,
    gan_d_n_layers=3,
    gan_d_lr=2e-4,
    gan_d_beta1=0.5,
    gan_loss_type="lsgan",
    gan_d_update_every=1,
    gan_shift_aware=True,
    in_ch=None,
    target_mode="map",
    loss_type="l1",
):
    """
    Misalignment-tolerant adversarial training.

    Generator G_A: uHSI cube -> FTIR map(s).
    Discriminator D_A: conditional PatchGAN. Receives a learned
        1x1-conv spectral projection of the uHSI input (8 channels)
        concatenated with either the ground-truth or the predicted
        FTIR. This is the standard pix2pix conditional formulation
        (Isola et al., 2017), adapted for spectral regression.

    Supervised data term:
        Shift-tolerant masked L1 (via MaskedRegressionLoss with the
        --max_shift search). This is the primary objective; the
        adversarial term sharpens the conditional output distribution.

    If gan_mode == 'pix2pix_inv':
        An auxiliary inverse generator G_B: FTIR -> uHSI is trained
        jointly, and an inverse-consistency reconstruction term
            L_inv = || G_B(G_A(x)) - x ||_1
        is added on masked cartilage pixels. This term is INSPIRED BY
        but distinct from CycleGAN's cycle consistency (Zhu et al.,
        2017): because our uHSI / FTIR data are paired, L_inv acts as
        a representation-preserving regulariser on paired data, not
        as an unsupervised domain-translation constraint. It is only
        applied when in_patch == out_patch (otherwise the inverse map
        at the border is ill-posed).

    Stabilisation:
        - LSGAN / hinge adversarial loss (no BCE by default).
        - Image-history replay buffer of 50 fakes for the D update
          (Shrivastava et al., 2017).
        - D updated every gan_d_update_every G updates.
        - No D_B: in paired settings it has no theoretical role
          and only destabilises training.

    Identity loss is intentionally NOT used: G(real_B) ≈ real_B is
    not well-defined when the input and output domains have different
    channel counts.
    """

    model.to(device)
    if scheduler_kwargs is None:
        scheduler_kwargs = {}

    # Supervised loss for TRAINING (shift-tolerant, with edge terms).
    # Uses the caller-provided loss_type so args.loss is respected in
    # the GAN path.
    sup_max_shift = max_shift if gan_shift_aware else 0
    sup_loss = MaskedRegressionLoss(
        loss_type=loss_type,
        smooth_weight=0.0,
        max_shift=sup_max_shift,
        huber_delta=huber_delta,
        y_bins=y_bins, y_weights=y_weights,
        grad_weight=grad_weight, lap_weight=lap_weight,
        var_weight=var_weight, var_kernel=var_kernel,
        fft_weight=fft_weight,
        spatial_loss_summary=spatial_loss_summary,
        spatial_loss_bands=spatial_loss_bands,
        per_channel_spatial=per_channel_spatial,
        spectral_smooth_weight=spectral_smooth_weight,
    ).to(device)



    # Honest validation loss: plain masked MAE, NO shift search,
    # NO edge-aware terms. This is the only metric BO ever sees.
    val_loss_fn = MaskedRegressionLoss(
        loss_type="l1",
        smooth_weight=0.0,
        max_shift=0,
        huber_delta=huber_delta,
        y_bins=None, y_weights=None,
        grad_weight=0.0, lap_weight=0.0,
        var_weight=0.0, var_kernel=var_kernel,
        fft_weight=0.0,
        spatial_loss_summary=spatial_loss_summary,
        spatial_loss_bands=spatial_loss_bands,
    ).to(device)


    # Probe one batch to learn shapes for building the discriminator
    x_probe, y_probe, m_probe = next(iter(train_loader))
    x_probe = x_probe.to(device)
    y_probe = y_probe.to(device)
    with torch.no_grad():
        p_probe = model(x_probe)

    # All 4D generator outputs in this codebase are BHWC; 3D outputs are
    # map-mode (B, P, P).
    gen_layout = None if p_probe.ndim == 3 else "bhwc"
    p4d = _to_4d_image(p_probe, layout=gen_layout)
    C_target = p4d.shape[1]

    # ---- Auto-fit PatchGAN depth to the actual output spatial size ----
    # The PatchGAN does: n_layers stride-2 4x4 convs, then two stride-1
    # 4x4 convs with padding=1 (each shrinks the map by 1 pixel).
    # We need spatial size > 1 after the stride-2 stack AND > kernel
    # after the stride-1 stack, otherwise PyTorch raises
    # "Kernel size can't be greater than actual input size".
    H_out, W_out = int(p4d.shape[2]), int(p4d.shape[3])
    S = min(H_out, W_out)

    def _max_safe_d_layers(spatial, requested):
        # After n stride-2 convs (k=4, s=2, p=1), size ~ ceil(S / 2^n).
        # We then need 2 more stride-1 4x4 convs with padding=1.
        # Each such conv requires input spatial >= 4.
        for n in range(requested, 0, -1):
            cur = spatial
            ok = True
            for _ in range(n):
                cur = (cur + 2 * 1 - 4) // 2 + 1   # exact stride-2 4x4 conv output
                if cur < 4:
                    ok = False
                    break
            # After the n stride-2 convs, we still need 2 stride-1 k=4 convs;
            # each requires the current spatial dim to be >= 4.
            if ok and cur >= 4:
                return n
        return 0

    n_layers_safe = _max_safe_d_layers(S, int(gan_d_n_layers))
    if n_layers_safe < 1:
        raise RuntimeError(
            f"[GAN] out_patch={S} is too small for any PatchGAN depth "
            f"(requested gan_d_n_layers={gan_d_n_layers}). "
            f"Increase out_patch or disable GAN for this trial."
        )
    if n_layers_safe != int(gan_d_n_layers):
        print(f"[GAN] Auto-reducing gan_d_n_layers from "
              f"{gan_d_n_layers} to {n_layers_safe} to fit "
              f"out_patch={S}.")
        gan_d_n_layers = n_layers_safe

    # Learned spectral projector for conditional D
    # (must be built BEFORE D_in_ch so the channel count is known).
    gan_proj_ch = 8
    proj_for_D = SpectralProjector(
        in_ch=x_probe.shape[1], n_proj=gan_proj_ch
    ).to(device)

    # Conditional D input: [proj(x), target_or_pred]
    D_in_ch = gan_proj_ch + C_target
    D_A = PatchGANDiscriminator(
        in_ch=D_in_ch, base_ch=gan_d_base_ch, n_layers=gan_d_n_layers
    ).to(device)


    # In the paired regime we use only D_A. CycleGAN's D_B operates on
    # unpaired data and has no role in the paired setting, so it is
    # omitted.
    G_B = None
    D_B = None
    if gan_mode == "pix2pix_inv":
        if in_ch is None:
            in_ch = x_probe.shape[1]
        G_B = InverseGeneratorUNet(
            in_ch=C_target, out_ch=in_ch,
            base_ch=max(16, gan_d_base_ch // 2),
            depth=3, dropout=0.1,
        ).to(device)
        # D_B left as None on purpose; see comment above.


    # Replay buffer for D update (image history; Shrivastava et al., 2017)
    buf_A = ImageBuffer(pool_size=50)

    # Add the spectral projector to the GENERATOR's parameter set
    # (it is conditioning learned jointly with G, not with D).
    g_params = list(model.parameters()) + list(proj_for_D.parameters())
    if G_B is not None:
        g_params += list(G_B.parameters())
    opt_G = torch.optim.Adam(g_params, lr=lr,
                             betas=(gan_d_beta1, 0.999))
    d_params = list(D_A.parameters())
    if D_B is not None:
        d_params += list(D_B.parameters())
    opt_D = torch.optim.Adam(d_params, lr=gan_d_lr,
                             betas=(gan_d_beta1, 0.999))
    
    # ---- LR scheduler for the generator (D kept at constant LR) ----
    if scheduler_type == "cosine":
        sched_G = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt_G,
            T_0=max(scheduler_kwargs.get("T_max", epochs) // 3, 5),
            T_mult=2,
            eta_min=scheduler_kwargs.get("eta_min", 1e-6),
        )
    elif scheduler_type == "plateau":
        sched_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_G, mode="min",
            factor=scheduler_kwargs.get("factor", 0.5),
            patience=scheduler_kwargs.get("patience", 5),
            threshold=scheduler_kwargs.get("threshold", 1e-4),
            min_lr=scheduler_kwargs.get("min_lr", 1e-6),
        )
    else:
        sched_G = None


    best_val_metric = float("inf")
    best_model_state = None
    epochs_no_improve = 0
    history = {
        "epoch": [], "train_loss": [], "val_loss": [], "val_select": [],
        "train_r2": [], "val_r2": [], "lr": [],
        "g_adv": [], "d_loss": [], "g_l1": [], "g_cycle": [],
    }

    def _inv_y(arr):
        if y_scaler is None or y_scaler_kind == "none":
            return arr
        try:
            flat = arr.reshape(-1, 1).astype(np.float64)
            # Clip for any scaler that can blow up on extrapolation
            if y_scaler_kind in ("log", "log_standard"):
                # expm1(50) ≈ 5.2e21, beyond any physical y; clip aggressively
                flat = np.clip(flat, -50.0, 50.0)
            elif y_scaler_kind in ("power", "quantile", "standard"):
                # These can extrapolate but won't blow up to inf as fast
                flat = np.clip(flat, -1e6, 1e6)

            finite = np.isfinite(flat).ravel()
            out = np.full_like(flat, np.nan)
            if finite.any():
                with np.errstate(over="ignore", invalid="ignore"):
                    inv = y_scaler.inverse_transform(flat[finite])
                out[finite] = inv
            # Replace any inf/nan in inverse result with nan (caller masks them)
            out = np.where(np.isfinite(out), out, np.nan)
            return out.reshape(arr.shape).astype(np.float32)
        except Exception:
            return arr


    step = 0
    for ep in range(1, epochs + 1):
        model.train()
        D_A.train()
        if G_B is not None: G_B.train()
        if D_B is not None: D_B.train()

        running = {"g_total": 0.0, "g_l1": 0.0, "g_adv": 0.0,
                   "g_cycle": 0.0, "d": 0.0, "n": 0}

        for x, y, m in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)

            # ---- Generator forward ----
            pred = model(x)
            if not torch.isfinite(pred).all():
                pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)

            # Determine the layout of the generator's spatial output.
            # - map mode + UNet2D       -> (B, P, P)         3D
            # - map mode + multi_unet   -> (B, P, P, n_maps) bhwc
            # - spectra mode + UNet/...  -> (B, P, P, C_out) bhwc
            if pred.ndim == 3:
                gen_layout = None
            else:
                gen_layout = "bhwc"   # all 4D producers in this codebase emit BHWC
            pred_4d = _to_4d_image(pred, layout=gen_layout)
            y_4d    = _to_4d_image(y,    layout=gen_layout)

            # ============== D update ==============
            if step % gan_d_update_every == 0:
                opt_D.zero_grad(set_to_none=True)
                # Build a DETACHED conditioning view for D so its backward()
                # does not consume the projector subgraph that G will need.
                with torch.no_grad():
                    x_cond_D = _summary_input_for_D(x, projector=proj_for_D)
                    if x_cond_D.shape[2:] != pred_4d.shape[2:]:
                        x_cond_D = F.interpolate(
                            x_cond_D, size=pred_4d.shape[2:],
                            mode="bilinear", align_corners=False,
                        )
                d_real = D_A(torch.cat([x_cond_D, y_4d.detach()], dim=1))
                # Image-buffer history for the fake half (standard pix2pix /
                # CycleGAN trick; stabilises D vs G dynamics).
                pred_4d_for_D = buf_A.query(pred_4d.detach())
                d_fake = D_A(torch.cat([x_cond_D, pred_4d_for_D], dim=1))

                d_loss = 0.5 * (
                    adversarial_loss(d_real, True, gan_loss_type)
                    + adversarial_loss(d_fake, False, gan_loss_type)
                )

                # D_B is intentionally omitted in the paired regime; it
                # has no theoretical role when input and output samples
                # are paired (CycleGAN's D_B operates on unpaired data).

                d_loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(D_A.parameters(), grad_clip)
                opt_D.step()
                running["d"] += d_loss.item() * x.size(0)

            # ============== G update ==============
            opt_G.zero_grad(set_to_none=True)
            # Build a FRESH conditioning view for G so the projector receives
            # gradients from the adversarial term.
            x_cond_G = _summary_input_for_D(x, projector=proj_for_D)
            if x_cond_G.shape[2:] != pred_4d.shape[2:]:
                x_cond_G = F.interpolate(
                    x_cond_G, size=pred_4d.shape[2:],
                    mode="bilinear", align_corners=False,
                )
            # Supervised, shift-tolerant data term (absorbs residual
            # spatial misalignment between paired uHSI and FTIR).
            g_l1 = sup_loss(pred, y, m)
            d_fake_for_g = D_A(torch.cat([x_cond_G, pred_4d], dim=1))

            g_adv = adversarial_loss(d_fake_for_g, True, gan_loss_type)
            g_total = gan_lambda_l1 * g_l1 + gan_lambda_adv * g_adv

            g_cycle_val = torch.tensor(0.0, device=device)
            if gan_mode == "pix2pix_inv" and G_B is not None:
                # The inverse-consistency reconstruction term is well-defined only
                # when the generator's output grid matches its input grid (i.e.
                # out_patch == patch). When out_patch < patch, the inverse mapping
                # at the cropped border is under-determined; the term is therefore
                # skipped in that case. See Methods, Loss Function.
                if pred_4d.shape[2:] == x.shape[2:]:
                    x_rec = G_B(pred_4d)

                    # 2D mask aligned to x (B, H, W).
                    m_sp = m
                    if m_sp.ndim == 4:
                        m_sp = (m_sp.any(dim=-1)
                                if m_sp.dtype == torch.bool
                                else (m_sp > 0).any(dim=-1))
                    if m_sp.ndim < 3:
                        # GAN cycle term requires a spatial mask. If we end
                        # up here, the user enabled GAN with a non-patch
                        # output head. Skip the cycle term cleanly.
                        m_sp = None
                    if m_sp is None:
                        cyc_sel = (x_rec - x).abs().mean()
                        g_cycle_val = cyc_sel
                        g_total = g_total + gan_lambda_cycle * g_cycle_val
                    else:
                        if tuple(m_sp.shape[1:]) != tuple(x.shape[2:]):
                            m_sp = (F.interpolate(
                                m_sp.float().unsqueeze(1),
                                size=x.shape[2:], mode="nearest",
                            ).squeeze(1) > 0.5)

                        m_x = m_sp.unsqueeze(1).expand_as(x)
                        if m_x.dtype != torch.bool:
                            m_x = m_x > 0

                        cyc = (x_rec - x).abs()
                        cyc_sel = cyc[m_x]
                        if cyc_sel.numel() > 0:
                            g_cycle_val = cyc_sel.mean()
                        g_total = g_total + gan_lambda_cycle * g_cycle_val

                    if tuple(m_sp.shape[1:]) != tuple(x.shape[2:]):
                        m_sp = (F.interpolate(
                            m_sp.float().unsqueeze(1),
                            size=x.shape[2:], mode="nearest",
                        ).squeeze(1) > 0.5)

                    m_x = m_sp.unsqueeze(1).expand_as(x)
                    if m_x.dtype != torch.bool:
                        m_x = m_x > 0

                    cyc = (x_rec - x).abs()
                    cyc_sel = cyc[m_x]
                    if cyc_sel.numel() > 0:
                        g_cycle_val = cyc_sel.mean()
                    g_total = g_total + gan_lambda_cycle * g_cycle_val
                else:
                    # patch != out_patch -> inverse map is ill-posed at the
                    # border; we report g_cycle_val = 0 in the training log.
                    pass

            # gan_lambda_identity is intentionally not used: see the
            # `--gan_lambda_identity` help text. Identity loss
            # G(real_B) ≈ real_B is undefined when input and output
            # domains have different channel counts (uHSI vs FTIR).
            if gan_lambda_identity > 0:
                import warnings
                warnings.warn(
                    f"gan_lambda_identity={gan_lambda_identity} is deprecated and "
                    f"will be ignored. See --help for details.",
                    DeprecationWarning, stacklevel=2,
                )



            if not torch.isfinite(g_total):
                # Skip this batch due to a non-finite generator loss.
                step += 1
                continue


            g_total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if G_B is not None:
                    torch.nn.utils.clip_grad_norm_(G_B.parameters(), grad_clip)
            opt_G.step()

            bs = x.size(0)
            running["g_total"] += g_total.item() * bs
            running["g_l1"]    += g_l1.item() * bs
            running["g_adv"]   += g_adv.item() * bs
            running["g_cycle"] += float(g_cycle_val) * bs
            running["n"]       += bs
            step += 1

        n = max(running["n"], 1)
        avg_g_total = running["g_total"] / n
        avg_g_l1    = running["g_l1"] / n
        avg_g_adv   = running["g_adv"] / n
        avg_g_cyc   = running["g_cycle"] / n
        avg_d       = running["d"] / n
        cur_lr      = opt_G.param_groups[0]["lr"]

        print(f"Epoch {ep}/{epochs}  G_total={avg_g_total:.3e}  "
              f"G_L1={avg_g_l1:.3e}  G_adv={avg_g_adv:.3e}  "
              f"G_cyc={avg_g_cyc:.3e}  D={avg_d:.3e}  lr={cur_lr:.2e}")

        # ---- Validation: use only the supervised shift-tolerant L1 ----
        avg_val_select = float("nan")
        val_r2 = float("nan")
        if val_loader is not None:
            model.eval()
            v_sum = 0.0; v_n = 0
            # Only the physical-units flat-R^2 accumulator is needed; it is
            # populated in MAP mode only. In SPECTRA mode val_r2_spectral
            # (per-pixel) is the canonical quantity.
            r2_acc_phys = {"n": 0, "sy": 0.0, "syy": 0.0, "sse": 0.0}
            val_blur_pen_sum = 0.0
            val_blur_n = 0
            # Spectra-mode per-pixel R^2 accumulators. Defined here in the
            # outer scope so the batch-loop updates and the post-loop
            # finalisation can both see them. Stays at (0, 0) -> NaN in
            # map mode, which is the correct behaviour for the BO contract.
            val_r2_spectral_sum = 0.0
            val_r2_spectral_n   = 0
            with torch.no_grad():

                for x, y, m in val_loader:
                    x = x.to(device); yd = y.to(device); md = m.to(device)
                    p = model(x)
                    v = val_loss_fn(p, yd, md)
                    if torch.isfinite(v):
                        v_sum += v.item() * x.size(0)
                        v_n   += x.size(0)
                    y_np = yd.cpu().numpy()
                    p_np = p.cpu().numpy()
                    m_np = md.cpu().numpy().astype(bool)

                    # Inverse to physical units once
                    y_np_phys = _inv_y(y_np)
                    p_np_phys = _inv_y(p_np)

                    # ---- Decomposed R^2 in spectra mode; flat R^2 in map mode ----
                    if y_np_phys.ndim == 4 and m_np.ndim == 3:
                        # Spectra patch output: (B, P, P, C_out), mask (B, P, P).
                        # Flatten spatial dims to (N_pixels, C_out) over masked pixels only.
                        B_, P1_, P2_, C_out_ = y_np_phys.shape
                        if C_out_ >= 2:
                            yt = y_np_phys.reshape(-1, C_out_)
                            yp = p_np_phys.reshape(-1, C_out_)
                            m_pix = m_np.reshape(-1).astype(bool)
                            row_valid = (
                                m_pix
                                & np.isfinite(yt).all(axis=1)
                                & np.isfinite(yp).all(axis=1)
                            )
                            if row_valid.any():
                                yt_v = yt[row_valid].astype(np.float64)
                                yp_v = yp[row_valid].astype(np.float64)
                                # Per-pixel spectral R^2 (the BO objective in spectra mode)
                                mu_pix = yt_v.mean(axis=1, keepdims=True)
                                ss_tot = np.sum((yt_v - mu_pix) ** 2, axis=1)
                                ss_res = np.sum((yt_v - yp_v) ** 2, axis=1)
                                keep = ss_tot > 1e-12
                                if keep.any():
                                    r2_pix = 1.0 - ss_res[keep] / ss_tot[keep]
                                    val_r2_spectral_sum += float(r2_pix.sum())
                                    val_r2_spectral_n   += int(keep.sum())

                    elif y_np_phys.ndim == 3 and m_np.ndim == 3:
                        # Map mode patch output (B, P, P) with spatial mask (B, P, P).
                        # Flat R^2 is meaningful here; reuse the original Welford-style accumulator.
                        sel = m_np & np.isfinite(y_np_phys) & np.isfinite(p_np_phys)
                        if sel.any():
                            yv = y_np_phys[sel].astype(np.float64)
                            pv = p_np_phys[sel].astype(np.float64)
                            r2_acc_phys["n"]   += yv.size
                            r2_acc_phys["sy"]  += yv.sum()
                            r2_acc_phys["syy"] += (yv * yv).sum()
                            r2_acc_phys["sse"] += ((yv - pv) ** 2).sum()

                    # ---- Physical-units blur penalty (channel-mean summary) ----
                    for i in range(x.size(0)):
                        if y_np_phys[i].ndim >= 2 and m_np[i].any():
                            ratio = local_std_ratio(
                                y_np_phys[i], p_np_phys[i], m_np[i], k=5
                            )
                            if np.isfinite(ratio):
                                val_blur_pen_sum += abs(1.0 - ratio)
                                val_blur_n += 1

            # ---- After the batch loop: finalise per-epoch aggregates ----
            if val_r2_spectral_n > 0:
                val_r2_spectral = val_r2_spectral_sum / val_r2_spectral_n
            else:
                val_r2_spectral = float("nan")
            history["val_r2_spectral"] = (
                history.get("val_r2_spectral", []) + [val_r2_spectral]
            )

            avg_val_select = v_sum / max(v_n, 1)


            def _r2_from_acc(acc):
                n = acc["n"]
                if n < 2:
                    return float("nan")
                mean_y = acc["sy"] / n
                ss_tot = acc["syy"] - n * mean_y * mean_y
                if ss_tot <= 0:
                    return float("nan")
                return 1.0 - acc["sse"] / ss_tot

            val_r2_phys = _r2_from_acc(r2_acc_phys)
            avg_val_blur_pen = val_blur_pen_sum / max(val_blur_n, 1)

            history["val_blur_pen"] = history.get("val_blur_pen", []) + [avg_val_blur_pen]

            # ---- Canonical val_r2 depends on target mode ----
            # In MAP mode r2_acc_phys was populated -> flat masked R^2 is valid.
            # In SPECTRA mode r2_acc_phys stays empty (-> nan) and we use the
            # per-pixel spectral R^2 instead (val_r2_spectral).
            is_spectra_mode = (
                val_r2_spectral_n > 0
                and np.isfinite(val_r2_spectral)
            )
            if is_spectra_mode:
                val_r2 = val_r2_spectral
                print(f"  Val: data-only (selection) = {avg_val_select:.3e}  "
                      f"val_R2(spectral, per-pixel) = {val_r2_spectral:.3f}  "
                      f"blur_pen = {avg_val_blur_pen:.3f}  "
                      f"[flat R2 not reported in spectra mode: misleading]")
            else:
                val_r2 = val_r2_phys
                print(f"  Val: data-only (selection) = {avg_val_select:.3e}  "
                      f"val_R2(physical) = {val_r2_phys:.3f}  "
                      f"blur_pen = {avg_val_blur_pen:.3f}")

            if np.isfinite(avg_val_select) and avg_val_select < best_val_metric:
                best_val_metric = avg_val_select
                _m = model.module if isinstance(model, nn.DataParallel) else model
                best_model_state = {k: v.cpu().clone()
                                    for k, v in _m.state_dict().items()}
                epochs_no_improve = 0
                print(f"  >> New best val = {avg_val_select:.3e}")
            else:
                epochs_no_improve += 1
            if patience_early_stop > 0 and epochs_no_improve >= patience_early_stop:
                print(f"  Early stopping at epoch {ep}")
                break

        if sched_G is not None:
            if scheduler_type == "plateau":
                if np.isfinite(avg_val_select):
                    sched_G.step(avg_val_select)
                else:
                    sched_G.step(avg_g_total)
            else:
                sched_G.step()

        history["epoch"].append(ep)
        history["train_loss"].append(avg_g_total)
        history["val_loss"].append(float("nan"))
        history["val_select"].append(avg_val_select)
        history["train_r2"].append(float("nan"))
        history["val_r2"].append(val_r2)
        history["lr"].append(cur_lr)
        history["g_adv"].append(avg_g_adv)
        history["d_loss"].append(avg_d)
        history["g_l1"].append(avg_g_l1)
        history["g_cycle"].append(avg_g_cyc)

    if best_model_state is not None:
        _m = model.module if isinstance(model, nn.DataParallel) else model
        _m.load_state_dict(best_model_state)
        print(f"Restored best model (val = {best_val_metric:.3e})")

    # Explicit cleanup of GAN-only networks and optimizers to prevent
    # accumulation across BO trials. The main `model` is retained for
    # use by the caller.
    try:
        del D_A
    except NameError:
        pass
    try:
        del D_B
    except NameError:
        pass
    try:
        del G_B
    except NameError:
        pass
    try:
        del opt_G, opt_D
    except NameError:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return history


def train_torch_model(model, train_loader, val_loader, device,
                      epochs=20, lr=1e-3,
                      scheduler_type="none", scheduler_kwargs=None,
                      grad_clip=1.0, patience_early_stop=15,
                      smooth_weight=0.01, max_shift=0,
                      loss_type="mse", huber_delta=1.0,
                      y_bins=None, y_weights=None,
                      grad_weight=0.0, lap_weight=0.0,
                      var_weight=0.0, var_kernel=5,
                      fft_weight=0.0,
                      spatial_loss_summary="mean", spatial_loss_bands=None,
                      y_scaler=None, y_scaler_kind="none",
                      per_channel_spatial=False,
                      spectral_smooth_weight=0.0):


    """
    Train with composite loss, select on data-only shift-aligned loss.
    """
    model.to(device)

    # Composite loss for OPTIMIZATION
    criterion = MaskedRegressionLoss(
        loss_type=loss_type,
        smooth_weight=smooth_weight,
        max_shift=max_shift,
        huber_delta=huber_delta,
        y_bins=y_bins,
        y_weights=y_weights,
        grad_weight=grad_weight,
        lap_weight=lap_weight,
        var_weight=var_weight,
        var_kernel=var_kernel,
        fft_weight=fft_weight,
        spatial_loss_summary=spatial_loss_summary,
        spatial_loss_bands=spatial_loss_bands,
        per_channel_spatial=per_channel_spatial,
        spectral_smooth_weight=spectral_smooth_weight,
    ).to(device)


    # Selection/validation metric. Deliberately NOT shift-aligned:
    # max_shift is a training-time alignment mechanism only. Allowing
    # the validation loss to search over shifts would bias BO toward
    # larger max_shift values, since the val score would underestimate
    # the true error. The selection metric is therefore the masked MAE
    # without any shift search.
    criterion_select = MaskedRegressionLoss(
        loss_type="l1",
        smooth_weight=0.0,
        max_shift=0,                   # honest, no alignment search
        huber_delta=huber_delta,       # unused for l1
        y_bins=None, y_weights=None,   # no per-bin reweighting
        grad_weight=0.0, lap_weight=0.0,
        var_weight=0.0, var_kernel=var_kernel,
        fft_weight=0.0,
        spatial_loss_summary=spatial_loss_summary,
        spatial_loss_bands=spatial_loss_bands,
    ).to(device)



    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    if scheduler_kwargs is None:
        scheduler_kwargs = {}

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(scheduler_kwargs.get("T_max", epochs) // 3, 5),
            T_mult=2,
            eta_min=scheduler_kwargs.get("eta_min", 1e-6),
        )
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=scheduler_kwargs.get("factor", 0.5),
            patience=scheduler_kwargs.get("patience", 5),
            threshold=scheduler_kwargs.get("threshold", 1e-4),
            min_lr=scheduler_kwargs.get("min_lr", 1e-6),
        )
    elif scheduler_type == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr * 10, epochs=epochs,
            steps_per_epoch=len(train_loader),
        )
    else:
        scheduler = None

    best_val_metric = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    history = {
        "epoch":      [],
        "train_loss": [],   # composite, sample-weighted
        "val_loss":   [],   # composite, sample-weighted
        "val_select": [],   # data-only, shift-aligned
        "train_r2":   [],   # masked R2 on scaled targets (training)
        "val_r2":     [],   # masked R2 on scaled targets (validation)
        "lr":         [],
    }



    def _r2_accum_init():
        return {"n": 0, "sy": 0.0, "syy": 0.0, "sse": 0.0}

    def _r2_accum_update(acc, y_np, p_np, m_np):
        """Update accumulator with masked (y, p) values from one batch."""
        m_np = m_np.astype(bool)
        if m_np.shape != y_np.shape:
            # broadcast spatial mask over channel axis if needed
            try:
                m_np = np.broadcast_to(m_np, y_np.shape)
            except Exception:
                return
        sel = m_np & np.isfinite(y_np) & np.isfinite(p_np)
        if not sel.any():
            return
        yv = y_np[sel].astype(np.float64)
        pv = p_np[sel].astype(np.float64)
        acc["n"]   += yv.size
        acc["sy"]  += yv.sum()
        acc["syy"] += (yv * yv).sum()
        acc["sse"] += ((yv - pv) ** 2).sum()

    def _r2_accum_value(acc):
        n = acc["n"]
        if n < 2:
            return float("nan")
        mean_y = acc["sy"] / n
        ss_tot = acc["syy"] - n * mean_y * mean_y   # = sum((y - mean_y)^2)
        if ss_tot <= 0:
            return float("nan")
        return 1.0 - acc["sse"] / ss_tot

    def _inv_y(arr):
        if y_scaler is None or y_scaler_kind == "none":
            return arr
        try:
            flat = arr.reshape(-1, 1).astype(np.float64)
            # Clip for any scaler that can blow up on extrapolation
            if y_scaler_kind in ("log", "log_standard"):
                # expm1(50) ≈ 5.2e21, beyond any physical y; clip aggressively
                flat = np.clip(flat, -50.0, 50.0)
            elif y_scaler_kind in ("power", "quantile", "standard"):
                # These can extrapolate but won't blow up to inf as fast
                flat = np.clip(flat, -1e6, 1e6)

            finite = np.isfinite(flat).ravel()
            out = np.full_like(flat, np.nan)
            if finite.any():
                with np.errstate(over="ignore", invalid="ignore"):
                    inv = y_scaler.inverse_transform(flat[finite])
                out[finite] = inv
            # Replace any inf/nan in inverse result with nan (caller masks them)
            out = np.where(np.isfinite(out), out, np.nan)
            return out.reshape(arr.shape).astype(np.float32)
        except Exception:
            return arr



    for ep in range(1, epochs + 1):
        _ep_t0 = time.time()
        # ---- Training (sample-weighted mean) ----
        model.train()
        train_loss_sum = 0.0
        train_n        = 0


        train_r2_acc = _r2_accum_init()
        train_r2_acc_u = _r2_accum_init()

        for batch_idx, (x, y, m) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)

            if not torch.isfinite(x).all():
                print(f"[ep {ep} batch {batch_idx}] Non-finite in X — skipping batch")
                continue
            if not torch.isfinite(y).all():
                print(f"[ep {ep} batch {batch_idx}] Non-finite in y — skipping batch")
                continue

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)

            if not torch.isfinite(pred).all():
                pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)

            loss = criterion(pred, y, m)
            if not torch.isfinite(loss):
                print(f"[ep {ep} batch {batch_idx}] Non-finite loss — skipping batch")
                continue

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            with torch.no_grad():
                y_np  = y.detach().cpu().numpy()
                p_np  = pred.detach().cpu().numpy()
                m_np  = m.detach().cpu().numpy()
                _r2_accum_update(train_r2_acc, y_np, p_np, m_np)
                # Mask out pixels where inverse y-scaling went non-finite, so
                # the unscaled R^2 is not biased toward the in-distribution
                # subset (otherwise non-linear y_scalers look artificially
                # good by silently dropping their failure pixels).
                y_np_phys = _inv_y(y_np)
                p_np_phys = _inv_y(p_np)
                finite_pair = np.isfinite(y_np_phys) & np.isfinite(p_np_phys)
                try:
                    m_eff_train = np.asarray(m_np, dtype=bool) & \
                                  np.broadcast_to(finite_pair, m_np.shape)
                except ValueError:
                    m_eff_train = m_np.astype(bool)
                _r2_accum_update(train_r2_acc_u,
                                 np.where(finite_pair, y_np_phys, 0.0),
                                 np.where(finite_pair, p_np_phys, 0.0),
                                 m_eff_train)


            bs = x.size(0)
            train_loss_sum += loss.item() * bs
            train_n        += bs

            if scheduler_type == "onecycle" and scheduler is not None:
                scheduler.step()

        avg_train_loss = (train_loss_sum / max(train_n, 1)) if train_n else float("nan")
        train_r2 = _r2_accum_value(train_r2_acc)
        train_r2_unscaled = _r2_accum_value(train_r2_acc_u)
        cur_lr = optimizer.param_groups[0]["lr"]
        _ep_dt = time.time() - _ep_t0
        print(f"Epoch {ep}/{epochs}  "
              f"train_loss = {avg_train_loss:.2e}  "
              f"train_R2(scaled) = {train_r2:.3f}  "
              f"train_R2(unscaled) = {train_r2_unscaled:.3f}  "
              f"lr = {cur_lr:.2e}  "
              f"time = {_ep_dt:.1f}s", flush=True)


        # ---- Validation (sample-weighted, both metrics) ----
        avg_val_loss   = float("nan")
        avg_val_select = float("nan")
        val_loss_for_sched = None

        early_stop_triggered = False

        val_r2 = float("nan")

        if val_loader is not None:
            model.eval()
            val_loss_sum   = 0.0
            val_select_sum = 0.0
            val_n          = 0
            val_r2_acc   = _r2_accum_init()
            val_r2_acc_u = _r2_accum_init()
            val_blur_pen_sum = 0.0
            val_blur_n = 0

            # Per-pixel R^2 accumulator for spectra-mode decomposition.
            # We accumulate sums over pixels of:
            #   r2_per_pixel_i = 1 - sum_c (y_ic - p_ic)^2 /
            #                        sum_c (y_ic - mean_c(y_ic))^2
            # and report the mean over valid (mask, non-degenerate) pixels.
            # Defined only when y_dev has a spectral axis (ndim==4 patch
            # mode (B,P,P,C_out) or ndim==2 per-pixel (B,C_out)).
            val_r2_spectral_sum = 0.0
            val_r2_spectral_n   = 0

            with torch.no_grad():
                for x, y, m in val_loader:
                    x     = x.to(device, non_blocking=True)
                    y_dev = y.to(device, non_blocking=True)
                    m_dev = m.to(device, non_blocking=True)
                    p     = model(x)

                    # v_loss: composite training criterion, kept ONLY for
                    # diagnostic logging in the history JSON. BO and the
                    # scheduler use v_select (honest masked MAE) instead.
                    v_loss = criterion(p, y_dev, m_dev)

                    # Masked MAE computed in PHYSICAL units so the BO
                    # objective is scaler-independent. Steps:
                    #  1) Inverse-scale y and p to physical units.
                    #  2) Replace non-finite entries with 0 to avoid
                    #     biasing v_select toward regions where the
                    #     inverse transform is well-defined.
                    #  3) Zero out the mask wherever either y_phys or
                    #     p_phys is non-finite, so the masked loss is
                    #     computed only where the inverse transform is
                    #     well-defined.
                    y_phys_np = _inv_y(y_dev.detach().cpu().numpy())
                    p_phys_np = _inv_y(p.detach().cpu().numpy())
                    finite_np = np.isfinite(y_phys_np) & np.isfinite(p_phys_np)
                    y_phys_np = np.where(finite_np, y_phys_np, 0.0).astype(np.float32)
                    p_phys_np = np.where(finite_np, p_phys_np, 0.0).astype(np.float32)
                    m_np_local = m_dev.detach().cpu().numpy().astype(bool)
                    if m_np_local.shape != finite_np.shape:
                        try:
                            m_eff = np.broadcast_to(m_np_local, finite_np.shape) & \
                                    np.broadcast_to(finite_np, finite_np.shape)
                        except ValueError:
                            m_eff = m_np_local
                    else:
                        m_eff = m_np_local & finite_np

                    y_dev_phys = torch.from_numpy(y_phys_np).to(device)
                    p_dev_phys = torch.from_numpy(p_phys_np).to(device)
                    m_dev_eff  = torch.from_numpy(m_eff).to(device)
                    v_select = criterion_select(p_dev_phys, y_dev_phys, m_dev_eff)


                    y_np = y_dev.detach().cpu().numpy()
                    p_np = p.detach().cpu().numpy()
                    m_np = m_dev.detach().cpu().numpy()
                    _r2_accum_update(val_r2_acc, y_np, p_np, m_np)
                    # Same correction as in train: exclude non-finite-inverse
                    # pixels from the mask, so non-linear y_scalers do not
                    # game val_r2_unscaled by failing silently outside their
                    # training distribution.
                    y_np_phys_acc = _inv_y(y_np)
                    p_np_phys_acc = _inv_y(p_np)
                    finite_pair_v = np.isfinite(y_np_phys_acc) & np.isfinite(p_np_phys_acc)
                    try:
                        m_eff_val = np.asarray(m_np, dtype=bool) & \
                                    np.broadcast_to(finite_pair_v, m_np.shape)
                    except ValueError:
                        m_eff_val = m_np.astype(bool)
                    _r2_accum_update(val_r2_acc_u,
                                     np.where(finite_pair_v, y_np_phys_acc, 0.0),
                                     np.where(finite_pair_v, p_np_phys_acc, 0.0),
                                     m_eff_val)

                    # ---- Spectra-mode per-pixel R^2 (val_r2_spectral) ----
                    # We compute it in PHYSICAL units (after inverse y-
                    # scaling), so the BO objective is scaler-independent
                    # across non-linear y_scalers, matching the convention
                    # for val_r2 / val_r2_minus_blur elsewhere.
                    y_np_phys = _inv_y(y_np)
                    p_np_phys = _inv_y(p_np)
                    if y_np_phys.ndim == 4 and m_np.ndim == 3:
                        # (B, P, P, C_out) spectra patch output.
                        B, P1, P2, C_out = y_np_phys.shape
                        if C_out >= 2:
                            yt = y_np_phys.reshape(-1, C_out)
                            yp = p_np_phys.reshape(-1, C_out)
                            mm = (
                                m_np[..., None]
                                .repeat(C_out, axis=-1)
                                .reshape(-1, C_out)
                                .astype(bool)
                            )
                            # Keep only pixels with full coverage and finite
                            # values; otherwise the per-pixel SS_tot is
                            # ill-defined.
                            row_valid = (
                                mm.all(axis=1)
                                & np.isfinite(yt).all(axis=1)
                                & np.isfinite(yp).all(axis=1)
                            )
                            if row_valid.any():
                                yt_v = yt[row_valid].astype(np.float64)
                                yp_v = yp[row_valid].astype(np.float64)
                                mu_pix = yt_v.mean(axis=1, keepdims=True)
                                ss_tot = np.sum((yt_v - mu_pix) ** 2, axis=1)
                                ss_res = np.sum((yt_v - yp_v) ** 2, axis=1)
                                keep = ss_tot > 1e-12
                                if keep.any():
                                    r2_pix = 1.0 - ss_res[keep] / ss_tot[keep]
                                    val_r2_spectral_sum += float(r2_pix.sum())
                                    val_r2_spectral_n   += int(keep.sum())
                    elif y_np_phys.ndim == 2 and m_np.ndim == 1:
                        # (B, C_out) per-pixel spectra output (no patch).
                        C_out = y_np_phys.shape[1]
                        if C_out >= 2:
                            row_valid = (
                                m_np.astype(bool)
                                & np.isfinite(y_np_phys).all(axis=1)
                                & np.isfinite(p_np_phys).all(axis=1)
                            )
                            if row_valid.any():
                                yt_v = y_np_phys[row_valid].astype(np.float64)
                                yp_v = p_np_phys[row_valid].astype(np.float64)
                                mu_pix = yt_v.mean(axis=1, keepdims=True)
                                ss_tot = np.sum((yt_v - mu_pix) ** 2, axis=1)
                                ss_res = np.sum((yt_v - yp_v) ** 2, axis=1)
                                keep = ss_tot > 1e-12
                                if keep.any():
                                    r2_pix = 1.0 - ss_res[keep] / ss_tot[keep]
                                    val_r2_spectral_sum += float(r2_pix.sum())
                                    val_r2_spectral_n   += int(keep.sum())
                    # else: map-mode tensor (no spectral axis); leave the
                    # accumulator at 0 / 0 -> NaN downstream.

                    
                    # Compute the blur penalty in PHYSICAL units (i.e.
                    # after inverse y-scaling). This is required so that
                    # the penalty is comparable across BO trials that
                    # pick different y_scalers, including non-linear ones
                    # (power, log, log_standard, quantile).
                    y_np_phys = _inv_y(y_np)
                    p_np_phys = _inv_y(p_np)
                    for i in range(x.size(0)):
                        if y_np_phys[i].ndim >= 2 and m_np[i].any():
                            ratio = local_std_ratio(
                                y_np_phys[i], p_np_phys[i], m_np[i], k=5
                            )
                            if np.isfinite(ratio):
                                val_blur_pen_sum += abs(1.0 - ratio)
                                val_blur_n += 1


                    bs = x.size(0)
                    if torch.isfinite(v_loss):
                        val_loss_sum += v_loss.item() * bs
                    if torch.isfinite(v_select):
                        val_select_sum += v_select.item() * bs
                        val_n          += bs

            if val_n > 0:
                avg_val_loss   = val_loss_sum   / val_n
                avg_val_select = val_select_sum / val_n

            val_r2_scaled   = _r2_accum_value(val_r2_acc)
            val_r2_unscaled = _r2_accum_value(val_r2_acc_u)
            avg_val_blur_pen = val_blur_pen_sum / max(val_blur_n, 1)
            history["val_blur_pen"] = history.get("val_blur_pen", []) + [avg_val_blur_pen]
            history["val_r2_scaled"] = history.get("val_r2_scaled", []) + [val_r2_scaled]

            if val_r2_spectral_n > 0:
                val_r2_spectral = val_r2_spectral_sum / val_r2_spectral_n
            else:
                val_r2_spectral = float("nan")
            history["val_r2_spectral"] = (
                history.get("val_r2_spectral", []) + [val_r2_spectral]
            )

            # ---- Canonical val_r2 depends on target mode ----
            # In MAP mode there is a single output channel, so flat R^2
            # over masked pixels is the natural quantity.
            # In SPECTRA mode the flat R^2 over (pixel x wavenumber)
            # conflates inter-pixel mean differences with intra-pixel
            # spectral shape fidelity (see r2_components_spectra docs),
            # so we use the per-pixel spectral R^2 instead.
            is_spectra_mode = (
                val_r2_spectral_n > 0
                and np.isfinite(val_r2_spectral)
            )
            if is_spectra_mode:
                val_r2 = val_r2_spectral
                print(f"  Val: composite = {avg_val_loss:.2e}  "
                      f"data-only (selection) = {avg_val_select:.2e}  "
                      f"val_R2(spectral, per-pixel) = {val_r2_spectral:.3f}  "
                      f"[flat R2 not reported in spectra mode: misleading]")
            else:
                val_r2 = val_r2_unscaled
                print(f"  Val: composite = {avg_val_loss:.2e}  "
                      f"data-only (selection) = {avg_val_select:.2e}  "
                      f"val_R2(scaled) = {val_r2_scaled:.3f}  "
                      f"val_R2(physical) = {val_r2_unscaled:.3f}")



            val_loss_for_sched = avg_val_select

            if np.isfinite(avg_val_select) and avg_val_select < best_val_metric:
                best_val_metric = avg_val_select
                _model = model.module if isinstance(model, nn.DataParallel) else model
                best_model_state = {
                    k: v.cpu().clone() for k, v in _model.state_dict().items()
                }
                epochs_no_improve = 0
                print(f"  >> New best val (data-only, shift-aligned) = {avg_val_select:.2e}")
            else:
                epochs_no_improve += 1

            if patience_early_stop > 0 and epochs_no_improve >= patience_early_stop:
                early_stop_triggered = True
        else:
            val_loss_for_sched = avg_train_loss


        # ---- Record this epoch ----
        history["epoch"].append(ep)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_select"].append(avg_val_select)
        history["train_r2"].append(train_r2)
        history["val_r2"].append(val_r2)
        history["lr"].append(cur_lr)


        # ---- Scheduler step ----
        if scheduler is not None and scheduler_type != "onecycle":
            if scheduler_type == "plateau":
                scheduler.step(
                    val_loss_for_sched if val_loss_for_sched is not None
                    else avg_train_loss
                )
            else:
                scheduler.step()

        if early_stop_triggered:
            print(f"  Early stopping at epoch {ep} "
                  f"(no improvement for {patience_early_stop} epochs)")
            break

    if best_model_state is not None:
        _model = model.module if isinstance(model, nn.DataParallel) else model
        _model.load_state_dict(best_model_state)
        print(f"Restored best model (val data-only = {best_val_metric:.2e})")
    else:
        print("WARNING: no best checkpoint saved (val never improved or "
              "val_loader=None). Using final-epoch weights.")

    return history

def _load_and_preprocess_one_npz(args_tuple):
    """
    Worker for parallel patch loading + per-sample SG smoothing + derivatives.
    Runs in a separate process. Returns dict of arrays for one sample.
    """
    (npz_path, name, map_field, smooth_x, sg_window, sg_poly,
     spectrum_norm, der_mode, delta_wl) = args_tuple

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float32)

    # Multi-map awareness
    if isinstance(map_field, (list, tuple)):
        y_list = [data[f"y_{mf}"] for mf in map_field]
        map_list = [data[f"map_data_{mf}"] for mf in map_field]
        y = np.stack(y_list, axis=-1).astype(np.float32)
        map_data = np.stack(map_list, axis=-1).astype(np.float32)
    else:
        y_key = f"y_{map_field}"
        m_key = f"map_data_{map_field}"
        if y_key in data:
            y = data[y_key]; map_data = data[m_key]
        else:
            y = data["y"]; map_data = data["map_data"]

    m_out = data["m_out"]
    coords = [tuple(c) for c in data["coords"].tolist()]
    mask = data["mask"]
    uhsi_shape = tuple(data["uhsi_shape"])
    X_aux = data["X_aux"] if "X_aux" in data.files else None

    # Preprocessing that is per-spectrum (SG smoothing, spectrum_norm,
    # derivatives) is deferred to the streaming DataLoader workers to
    # keep peak RAM bounded. This flag is retained to allow disabling
    # the deferral for ablations; the current runs always defer.
    _defer_to_stream = True

    if not _defer_to_stream:
        if smooth_x:
            X = apply_savgol_smoothing(X, window=sg_window, polyorder=sg_poly)
        if spectrum_norm != "none":
            X = apply_spectrum_norm(X, mode=spectrum_norm)
        if der_mode != "raw":
            X = apply_derivatives(X, der_mode=der_mode,
                                sg_window=sg_window,
                                polyorder=sg_poly,
                                delta=delta_wl)

    if X_aux is None:
        if X.ndim == 4:
            X_aux = np.zeros((X.shape[0], X.shape[1], X.shape[2], 0),
                             dtype=np.float32)
        else:
            X_aux = np.zeros((X.shape[0], 0), dtype=np.float32)
    else:
        X_aux = X_aux.astype(np.float32)

    return {
        "name": name,
        "X": X, "y": y, "m_out": m_out, "coords": coords,
        "X_aux": X_aux, "map_data": map_data, "mask": mask,
        "uhsi_shape": uhsi_shape,
    }

def _load_and_preprocess_one_mat(args_tuple):
    """
    Worker for parallel raw .mat loading and patch extraction.

    Per-spectrum preprocessing (SG smoothing, spectrum_norm,
    derivatives) is deferred to the streaming DataLoader workers
    (UHsiStreamingDataset). The eager block is retained for
    ablations, gated by ``_defer_to_stream``. In the current design
    ``_defer_to_stream`` is True, so X is returned with the raw
    channel count. This avoids a 2x-3x channel blow-up in RAM when
    BO selects der_mode in {der1, der2, raw+der1+der2, ...}.
    """

    (s, args_dict, wls, overlap_override) = args_tuple

    # Reconstruct a lightweight Namespace from a dict (argparse Namespaces
    # don't pickle reliably across processes on all systems; dict is safe).
    args_ns = argparse.Namespace(**args_dict)

    X, y, m_out, m_in, coords, gt, mask_s, shape, ftir_wls, X_aux = \
        load_and_extract_one_sample(
            s["ftir_path"], s["uhsi_path"], args_ns,
            wavelengths=wls,
            overlap_override=overlap_override,
            ftir_cube_path=s.get("ftir_cube_path"),
        )

    # Defer SG / SNV / derivatives to the streaming DataLoader workers,
    # to keep peak RAM bounded in spectra-mode BO. Numerically identical
    # to the eager path because all three operations are per-spectrum.
    _defer_to_stream = True

    if not _defer_to_stream:
        if args_ns.smooth_x and X.size > 0:
            X = apply_savgol_smoothing(
                X, window=args_ns.sg_window, polyorder=args_ns.sg_poly,
            )
        if args_ns.spectrum_norm != "none" and X.size > 0:
            X = apply_spectrum_norm(X, mode=args_ns.spectrum_norm)
        if args_ns.der_mode != "raw" and X.size > 0:
            delta = float(wls[1] - wls[0]) if len(wls) > 1 else 1.0
            X = apply_derivatives(X, der_mode=args_ns.der_mode,
                                sg_window=args_ns.sg_window,
                                polyorder=args_ns.sg_poly,
                                delta=delta)
    import tempfile
    tmp_dir = args_dict.get("_ipc_tmp_dir") or tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f"_ipc_{s['name']}_{os.getpid()}.npz")
    np.savez(
        tmp_path,
        X=X, y=y, m_out=m_out, m_in=m_in,
        gt=gt, mask=mask_s,
        X_aux=(X_aux if X_aux is not None else np.zeros((0,), dtype=np.float32)),
    )
    return {
        "name": s["name"],
        "tmp_path": tmp_path,
        "coords": coords,
        "shape": shape,
        "ftir_wls": ftir_wls,
        "ftir_cube_path": s.get("ftir_cube_path"),
        "has_aux": X_aux is not None,
    }


def load_split_parallel_mat(sample_list, split_name, args, wls,
                            overlap_override=None, C_total=None,
                            n_workers=None):
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()

    if n_workers is None:
        if slurm_cpus.isdigit():
            n_workers = max(int(slurm_cpus) - 1, 1)
        else:
            n_workers = max(os.cpu_count() - 1, 1)

    if bool(getattr(args, "_is_bo_trial", False)):
        slurm_cpus_int = int(slurm_cpus) if slurm_cpus.isdigit() else 0
        bo_cap = 2 if slurm_cpus_int and slurm_cpus_int <= 4 else 4
        if n_workers > bo_cap:
            print(f"[parallel-mat] BO trial: capping n_workers from "
                  f"{n_workers} to {bo_cap}", flush=True)
            n_workers = bo_cap

    n_workers = max(1, min(n_workers, len(sample_list))) if sample_list else 1


    # argparse.Namespace -> dict (picklable, no surprises across processes)
    args_dict = vars(args)

    tasks = [(s, args_dict, wls, overlap_override) for s in sample_list]
    name_order = [s["name"] for s in sample_list]

    print(f"[parallel-mat] {split_name}: {len(tasks)} samples, "
          f"{n_workers} workers")

    results = {}
    if not tasks:
        # Empty split: fall through to placeholder construction below
        pass
    else:
        t_pool_start = time.time()
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=_MP_CTX_SPAWN,
        ) as ex:
            futures = {
                ex.submit(_load_and_preprocess_one_mat, t): t[0]["name"]
                for t in tasks
            }
            for fut in as_completed(futures):
                name = futures[fut]
                meta = fut.result()
                # mmap the worker output — zero-copy
                with np.load(meta["tmp_path"], mmap_mode="r") as d:
                    results[name] = {
                        "X":     np.array(d["X"],     copy=True),
                        "y":     np.array(d["y"],     copy=True),
                        "m_out": np.array(d["m_out"], copy=True),
                        "m_in":  np.array(d["m_in"],  copy=True),
                        "gt":    np.array(d["gt"],    copy=True),
                        "mask":  np.array(d["mask"],  copy=True),
                        "X_aux": (np.array(d["X_aux"], copy=True)
                                if meta["has_aux"] else None),
                        "coords":         meta["coords"],
                        "shape":          meta["shape"],
                        "ftir_wls":       meta["ftir_wls"],
                        "ftir_cube_path": meta["ftir_cube_path"],
                        "name":           name,
                        "_tmp_path":      meta["tmp_path"],
                    }

        print(f"[parallel-mat] {split_name} pool finished in "
              f"{time.time() - t_pool_start:.1f}s", flush=True)


    if results:
        name_keep = [n for n in name_order if n in results]
        total_N = sum(results[n]["X"].shape[0] for n in name_keep)
        first = results[name_keep[0]]

        X_cat = np.empty((total_N,) + first["X"].shape[1:], dtype=np.float32)
        y_cat = np.empty((total_N,) + first["y"].shape[1:], dtype=first["y"].dtype)
        m_cat = np.empty((total_N,) + first["m_out"].shape[1:], dtype=np.bool_)
        m_in_cat = np.empty((total_N,) + first["m_in"].shape[1:], dtype=np.bool_)
        aux_C = first["X_aux"].shape[-1] if first["X_aux"] is not None else 0
        X_aux_cat = np.empty(
            (total_N,) + first["X"].shape[1:3] + (aux_C,), dtype=np.float32
        ) if first["X"].ndim == 4 else np.empty((total_N, aux_C), dtype=np.float32)

        coords_all, meta_list = [], []
        idx = 0
        for name in name_keep:
            r = results[name]
            n = r["X"].shape[0]
            X_cat[idx:idx+n] = r["X"]
            y_cat[idx:idx+n] = r["y"]
            m_cat[idx:idx+n] = r["m_out"]
            m_in_cat[idx:idx+n] = r["m_in"]
            if r["X_aux"] is not None and aux_C > 0:
                X_aux_cat[idx:idx+n] = r["X_aux"]
            coords_all.extend(r["coords"])
            # During BO trials, drop the heavy map_data field — it's only
            # used for visualization, which BO trials skip. Saves ~5 MB per
            # sample for full_loso.
            meta_entry = {
                "name": r["name"], "mask": r["mask"],
                "shape": r["shape"], "n_patches": n, "coords": r["coords"],
                "ftir_wls": r["ftir_wls"], "ftir_cube_path": r["ftir_cube_path"],
            }
            if not bool(getattr(args, "_is_bo_trial", False)):
                meta_entry["map_data"] = r["gt"]
            else:
                meta_entry["map_data"] = None
            meta_list.append(meta_entry)

            idx += n
            # Drop reference so mmap can be unmapped
            r["X"] = r["y"] = r["m_out"] = r["m_in"] = r["X_aux"] = None

        # Force garbage collection BEFORE removing files so mmap views are released
        gc.collect()

        # Cleanup temp files (on Windows, mmap must be closed before unlink)
        for r in results.values():
            tp = r.get("_tmp_path")
            if tp and os.path.exists(tp):
                try:
                    os.remove(tp)
                except Exception as e:
                    print(f"[parallel-mat] could not remove {tp}: {e}")

        results.clear()
        gc.collect()

    else:
        # Empty split placeholders (mirror the empty branch in original load_split)
        if args.avg_mode == "patch":
            X_cat = np.empty((0, args.patch, args.patch, C_total or 0),
                             dtype=np.float32)
        else:
            X_cat = np.empty((0, C_total or 0), dtype=np.float32)
        if args.target_mode == "map":
            if args.out_avg_mode == "patch":
                y_cat = np.empty((0, args.out_patch, args.out_patch),
                                 dtype=np.float32)
            elif args.out_avg_mode == "flat":
                y_cat = np.empty((0, args.out_patch * args.out_patch),
                                 dtype=np.float32)
            else:
                y_cat = np.empty((0,), dtype=np.float32)
        else:
            # Spectra mode: use the correct spectral channel count so downstream
            # concatenation with real spectra data does not fail.
            C_ftir_est = int(FTIR_WLS_OVERRIDE.size)  # 623 by construction
            if args.out_avg_mode == "patch":
                y_cat = np.empty((0, args.out_patch, args.out_patch, C_ftir_est),
                                 dtype=np.float32)
            elif args.out_avg_mode == "flat":
                y_cat = np.empty((0, args.out_patch * args.out_patch * C_ftir_est),
                                 dtype=np.float32)
            else:
                y_cat = np.empty((0, C_ftir_est), dtype=np.float32)
        m_cat = np.empty((0, args.out_patch, args.out_patch), dtype=np.bool_)
        m_in_cat = np.empty((0, args.patch, args.patch), dtype=np.bool_)
        X_aux_cat = np.empty((0, args.patch, args.patch, 0), dtype=np.float32)

    return X_cat, y_cat, m_cat, m_in_cat, coords_all, meta_list, X_aux_cat


def load_patches_from_npz(out_dir, sample_list, split_name,
                          use_test_patches=False, map_field="AmideI"):
    """
    Load pre-extracted patches from .npz files.

    Parameters:
        out_dir:            Directory containing the .npz files
        sample_list:        List of sample dicts with 'name' key
        split_name:         "train", "val", or "test" (for logging)
        use_test_patches:   If True, load *_test.npz; else *_train.npz
        map_field:          Which FTIR map to use as ground truth (e.g. "AmideI")

    Returns:
        X_cat, y_cat, m_cat, coords_all, meta_list
    """
    X_all, y_all, m_all, coords_all = [], [], [], []
    aux_all = []
    meta_list = []

    suffix = "_test.npz" if use_test_patches else "_train.npz"


    for s in sample_list:
        name = s["name"]
        npz_path = os.path.join(out_dir, f"{name}{suffix}")

        # Fall back to _train.npz if _test.npz doesn't exist
        if not os.path.exists(npz_path):
            npz_path_alt = os.path.join(out_dir, f"{name}_train.npz")
            if os.path.exists(npz_path_alt):
                npz_path = npz_path_alt
            else:
                print(f"  WARNING: No .npz file found for {name}, skipping.")
                continue

        print(f"Loading {split_name}: {name} from {os.path.basename(npz_path)} ...")
        data = np.load(npz_path, allow_pickle=True)

        X = data["X"]

        # Multi-map support: load list of maps if args.map_fields is set,
        # otherwise legacy single-map format (y, map_data) or per-field
        # (y_<field>, map_data_<field>).
        if isinstance(map_field, (list, tuple)):
            y_list_local = []
            map_data_list = []
            for mf in map_field:
                y_k = f"y_{mf}"
                m_k = f"map_data_{mf}"
                if y_k not in data:
                    raise KeyError(
                        f"Multi-map: field '{mf}' missing in {npz_path}. "
                        f"Available y_* keys: "
                        f"{[k for k in data.files if k.startswith('y_')]}"
                    )
                y_list_local.append(data[y_k])
                map_data_list.append(data[m_k])
            y = np.stack(y_list_local, axis=-1).astype(np.float32)
            map_data = np.stack(map_data_list, axis=-1).astype(np.float32)
        else:
            y_key = f"y_{map_field}"
            map_key = f"map_data_{map_field}"
            if y_key in data:
                y = data[y_key]
                map_data = data[map_key]
            elif "y" in data:
                y = data["y"]
                map_data = data["map_data"]
            else:
                available = [k for k in data.files if k.startswith("y_")]
                raise KeyError(
                    f"Map field '{map_field}' not found in {npz_path}. "
                    f"Available: {available}"
                )

        m_out = data["m_out"]
        coords = data["coords"].tolist()
        coords = [tuple(c) for c in coords]
        mask = data["mask"]
        uhsi_shape = tuple(data["uhsi_shape"])

        # Optional aux channels saved at patch-extraction time.
        # If absent, we will rebuild a zero-channel placeholder below.
        X_aux = data["X_aux"] if "X_aux" in data.files else None

        print(f"  X: {X.shape}, y: {y.shape}, patches: {X.shape[0]}, map: {map_field}")

        X_all.append(X)
        y_all.append(y)
        m_all.append(m_out)
        coords_all.extend(coords)
        aux_all.append(
            X_aux.astype(np.float32) if X_aux is not None
            else np.zeros((X.shape[0], X.shape[1], X.shape[2], 0),
                          dtype=np.float32)
            if X.ndim == 4
            else np.zeros((X.shape[0], 0), dtype=np.float32)
        )
        meta_list.append({
            "name": name,
            "map_data": map_data,
            "mask": mask,
            "shape": uhsi_shape,
            "n_patches": X.shape[0],
            "coords": coords,
        })

        del data
        gc.collect()


    if X_all:
        X_cat = np.concatenate(X_all, axis=0); del X_all
        y_cat = np.concatenate(y_all, axis=0); del y_all
        m_cat = np.concatenate(m_all, axis=0); del m_all
        X_aux_cat = np.concatenate(aux_all, axis=0) if aux_all else None
        del aux_all
        gc.collect()
    else:
        raise RuntimeError(f"No data loaded for split '{split_name}'")

    return X_cat, y_cat, m_cat, coords_all, meta_list, X_aux_cat


def load_patches_from_npz_parallel(
    out_dir, sample_list, split_name, args, wls,
    use_test_patches=False, map_field="AmideI", n_workers=None,
):
    """
    Parallel version: loads .npz files and applies SG/spectrum_norm/derivatives
    inside worker processes. SG/derivatives are MOVED OUT of run_one_fold and
    done here per-sample, so the main process is freed up.
    """
    if n_workers is None:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
        if slurm_cpus.isdigit():
            n_workers = max(int(slurm_cpus) - 1, 1)
        else:
            n_workers = max(os.cpu_count() - 1, 1)

    suffix = "_test.npz" if use_test_patches else "_train.npz"

    tasks = []
    name_order = []
    for s in sample_list:
        name = s["name"]
        npz_path = os.path.join(out_dir, f"{name}{suffix}")
        if not os.path.exists(npz_path):
            alt = os.path.join(out_dir, f"{name}_train.npz")
            if os.path.exists(alt):
                npz_path = alt
            else:
                print(f"  WARNING: No .npz file for {name}, skipping.")
                continue
        delta_wl = float(wls[1] - wls[0]) if len(wls) > 1 else 1.0
        tasks.append((
            npz_path, name, map_field,
            args.smooth_x, args.sg_window, args.sg_poly,
            args.spectrum_norm, args.der_mode, delta_wl,
        ))
        name_order.append(name)

    print(f"[parallel-load] {split_name}: {len(tasks)} samples, "
          f"{n_workers} workers")

    results = {}
    t_pool_start = time.time()
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=_MP_CTX_SPAWN,
    ) as ex:
        futures = {ex.submit(_load_and_preprocess_one_npz, t): t[1] for t in tasks}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results[name] = fut.result()
                print(f"  [parallel-load] done: {name} "
                      f"(elapsed {time.time() - t_pool_start:.1f}s)",
                      flush=True)
            except Exception as e:
                print(f"  [parallel-load] FAILED {name}: {e}", flush=True)
                raise
    print(f"[parallel-load] {split_name} pool finished in "
          f"{time.time() - t_pool_start:.1f}s", flush=True)


    # Concatenate in deterministic order
    X_all, y_all, m_all, coords_all, aux_all, meta_list = [], [], [], [], [], []
    for name in name_order:
        if name not in results:
            continue
        r = results[name]
        X_all.append(r["X"])
        y_all.append(r["y"])
        m_all.append(r["m_out"])
        coords_all.extend(r["coords"])
        aux_all.append(r["X_aux"])
        meta_list.append({
            "name": name,
            "map_data": r["map_data"],
            "mask": r["mask"],
            "shape": r["uhsi_shape"],
            "n_patches": r["X"].shape[0],
            "coords": r["coords"],
        })

    if not X_all:
        raise RuntimeError(f"No data loaded for split '{split_name}'")

    X_cat = np.concatenate(X_all, axis=0); del X_all
    y_cat = np.concatenate(y_all, axis=0); del y_all
    m_cat = np.concatenate(m_all, axis=0); del m_all
    X_aux_cat = np.concatenate(aux_all, axis=0); del aux_all
    gc.collect()
    return X_cat, y_cat, m_cat, coords_all, meta_list, X_aux_cat


def discover_sample_files(data_dir, target_mode="map"):
    ftir_files, uhsi_files, cube_files = {}, {}, {}
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if fname.endswith("_ftir_maps.mat"):
            ftir_files[fname[:-len("_ftir_maps.mat")]] = fpath
        elif fname.endswith("_uhsi.mat"):
            uhsi_files[fname[:-len("_uhsi.mat")]] = fpath
        elif fname.endswith("_ftir.mat"):
            cube_files[fname[:-len("_ftir.mat")]] = fpath

    samples = []
    for key in sorted(ftir_files.keys()):
        if key not in uhsi_files:
            print(f"Warning: '{key}_ftir_maps.mat' found but no matching "
                  f"'{key}_uhsi.mat'; skipping.")
            continue

        if target_mode == "spectra" and key not in cube_files:
            print(f"Warning: spectra mode but no _ftir.mat for {key}")
            continue
        samples.append({
            "name": key,
            "prefix": key.split("_")[0],
            "ftir_path": ftir_files[key],         # maps file (always)
            "uhsi_path": uhsi_files[key],
            "ftir_cube_path": cube_files.get(key),  # only used in spectra
        })
    return samples

def discover_samples_from_npz(patch_dir):
    """
    Discover sample names from .npz files in patch_dir.
    Returns a list of dicts compatible with discover_sample_files output,
    but without ftir_path / uhsi_path (not needed for patch-based loading).
    """
    seen = {}
    for fname in sorted(os.listdir(patch_dir)):
        if not fname.endswith(".npz"):
            continue
        # Strip _train.npz or _test.npz to get sample name
        for suffix in ("_train.npz", "_test.npz"):
            if fname.endswith(suffix):
                name = fname[: -len(suffix)]
                if name not in seen:
                    prefix = name.split("_")[0]
                    seen[name] = {
                        "name": name,
                        "prefix": prefix,
                        "ftir_path": None,
                        "uhsi_path": None,
                        "ftir_cube_path": None,
                    }

                break
    return list(seen.values())

def extract_aux_patches(aux_dict, coords, in_patch):
    """
    Build (N, P, P, K_total) aux tensor by combining:
      - per-image (H, W) maps that get cropped at each (r, c)
      - per-sample (P, P) maps that are reused for every patch
    """
    if aux_dict is None:
        return None
    full_maps = aux_dict.get("full_maps", [])
    patch_maps = aux_dict.get("patch_maps", [])
    if not full_maps and not patch_maps:
        return None

    P = in_patch
    rad = P // 2
    N = len(coords)
    K_full = len(full_maps)
    K_patch = len(patch_maps)
    K_total = K_full + K_patch

    out = np.zeros((N, P, P, K_total), dtype=np.float32)

    # Per-image maps: stack and crop all patches at once via fancy indexing.
    # This eliminates the per-coord Python loop overhead.
    if full_maps:
        rs = np.array([r for r, c in coords], dtype=np.intp)
        cs = np.array([c for r, c in coords], dtype=np.intp)
        H_m, W_m = full_maps[0].shape
        # Filter coords that fit entirely within the image
        in_bounds = ((rs - rad >= 0) & (cs - rad >= 0)
                     & (rs + rad + 1 <= H_m) & (cs + rad + 1 <= W_m))
        valid_idx = np.where(in_bounds)[0]
        if valid_idx.size > 0:
            rs_v = rs[valid_idx]
            cs_v = cs[valid_idx]
            # Build row/col index arrays once and broadcast
            row_off = np.arange(-rad, rad + 1, dtype=np.intp)  # (P,)
            col_off = np.arange(-rad, rad + 1, dtype=np.intp)  # (P,)
            row_idx = rs_v[:, None, None] + row_off[None, :, None]  # (N_v, P, 1)
            col_idx = cs_v[:, None, None] + col_off[None, None, :]  # (N_v, 1, P)
            for k, m in enumerate(full_maps):
                # Same H_m, W_m assumption holds for all full_maps
                out[valid_idx, :, :, k] = m[row_idx, col_idx]



    # Per-sample maps -> broadcast identical content to all patches
    for k, m in enumerate(patch_maps):
        if m.shape != (P, P):
            raise ValueError(
                f"patch_maps entry must be ({P},{P}); got {m.shape}"
            )
        out[:, :, :, K_full + k] = m[None, :, :]

    return out



def load_and_extract_one_sample(
    ftir_path, uhsi_path, args, wavelengths=None, overlap_override=None,
    ftir_cube_path=None,
):


    """
    Load one FTIR + uHSI pair, apply mask and baseline correction,
    extract patches from the FULL image (no column split).

    Returns:
        X: array of input patches
        y: array of output targets
        coords: list of (row, col)
        map_data: (H, W) ground truth map
        mask: (H, W) boolean cartilage mask
        uhsi_shape: (H, W, C)
    """

    # ---- Load FTIR maps (v7.3 aware) ----
    ftir = load_mat_any(ftir_path)
    R = ftir[args.amide_struct_key]

    amide_struct = _get_field(R, args.amide_field)
    amide1 = np.asarray(
        _get_field(amide_struct, args.amide_map_field), dtype=np.float32)

    if args.target_mode == "map":
        # Multi-map awareness: when --map_fields is provided AND we are in
        # multi_unet mode, stack all requested maps along the last axis to
        # produce a (H, W, n_maps) target. Otherwise behave as before.
        use_multi = (getattr(args, "map_fields", None)
                     and getattr(args, "model", "") == "multi_unet")
        if use_multi:
            gt_list = []
            for mf in args.map_fields:
                ms = _get_field(R, mf)
                gt_list.append(
                    np.asarray(_get_field(ms, args.amide_map_field),
                               dtype=np.float32)
                )
            gt = np.stack(gt_list, axis=-1).astype(np.float32)  # (H, W, n_maps)
            del gt_list
        else:
            map_struct = _get_field(R, args.map_field)
            gt = np.asarray(_get_field(map_struct, args.amide_map_field),
                            dtype=np.float32)
        ftir_wls = None
    else:  # spectra
        if ftir_cube_path is None:
            raise FileNotFoundError(f"target_mode=spectra requires _ftir.mat for {uhsi_path}")
        with h5py.File(ftir_cube_path, "r") as f:
            gt = f[args.ftir_key][()].astype(np.float32).transpose(2, 1, 0)  # (H, W, C_ftir)

        # ---- Use the hardcoded wavenumber axis instead of R.WN ----
        ftir_wls = FTIR_WLS_OVERRIDE.astype(np.float32)

        # Sanity check: axis length must match the spectral dimension of the cube.
        if ftir_wls.size != gt.shape[2]:
            raise ValueError(
                f"Hardcoded FTIR_WLS_OVERRIDE has {ftir_wls.size} points "
                f"but FTIR cube has {gt.shape[2]} spectral channels for "
                f"{ftir_cube_path}. Either regenerate FTIR_WLS_OVERRIDE "
                f"with the correct length, or check that the cube is "
                f"transposed correctly (expected order: H, W, C)."
            )

    # NOTE: do NOT delete R yet — we still need it for cartilage_mask below.
    del ftir, amide_struct
    if args.target_mode == "map" and "map_struct" in locals():
        del map_struct
    gc.collect()



    # Make a writable copy because downstream baseline correction is in-place-adjacent
    uhsi = _load_raw_cube_cached(uhsi_path, args.uhsi_key).copy()

    H, W, C = uhsi.shape
    print(f"  uHSI shape: {uhsi.shape}, map shape: {gt.shape}")

    # ---- Wavelengths ----
    if wavelengths is None:
        wavelengths = np.linspace(
            getattr(args, "uhsi_wl_min", 397.01),
            getattr(args, "uhsi_wl_max", 1004.52),
            C,
        ).astype(np.float32)

    # ---- Optional wavelength exclusion (before any preprocessing) ----
    exclude_ranges = getattr(args, "exclude_wl_ranges", None)
    if exclude_ranges:
        uhsi, wavelengths, _ = apply_wavelength_exclusion(
            uhsi, wavelengths, exclude_ranges,
        )
        C = uhsi.shape[2]
        print(f"  After wavelength exclusion: uHSI shape={uhsi.shape}")


    # ---- Load BOTH masks ----
    # final_mask: used for patch filtering, targets, and visualization (as before)
    # cartilage_mask: used ONLY for the depth auxiliary channel
    if _has_field(R, "final_mask"):
        mask = np.asarray(_get_field(R, "final_mask")).astype(bool)
        print(f"  Cartilage pixels (saved final_mask): {mask.sum()}")
    elif _has_field(R, "cartilage_mask"):
        mask = np.asarray(_get_field(R, "cartilage_mask")).astype(bool)
        print(f"  Cartilage pixels (saved cartilage_mask as fallback): {mask.sum()}")
    else:
        if args.amide_thresh <= 0.0:
            raise ValueError(
                f"Sample at {ftir_path} lacks both final_mask and "
                f"cartilage_mask, AND --amide_thresh<=0. This would silently "
                f"label every positive pixel as cartilage, biasing all "
                f"downstream statistics. Either provide a saved mask in "
                f"the .mat file, or set --amide_thresh to a positive value."
            )
        print("  WARNING: no final_mask/cartilage_mask; falling back to AmideI threshold")
        mask = make_cartilage_mask(
            amide1, thresh=args.amide_thresh, min_size=args.min_obj_size)

    # Separate mask used exclusively for depth-channel computation
    if _has_field(R, "cartilage_mask"):
        depth_mask = np.asarray(_get_field(R, "cartilage_mask")).astype(bool)
        print(f"  Depth mask pixels (cartilage_mask): {depth_mask.sum()}")
    else:
        depth_mask = mask  # fallback: use the same mask
        print("  WARNING: no cartilage_mask found; using final_mask for depth too")


    del R
    gc.collect()

    # ---- Spatial downsampling (uHSI, mask, amide1, map_data) ----
    scale = getattr(args, "spatial_scale", 1.0)
    ds_method = getattr(args, "ds_method", "area")
    if scale < 1.0:
        # uHSI/mask/amide as before
        if ds_method == "area":
            uhsi, mask, amide1 = spatial_downsample_area(uhsi, mask, amide1, scale)
            H_new, W_new = uhsi.shape[:2]
            # Downsample depth_mask using nearest-neighbor logic (same as mask)
            depth_mask = resize(depth_mask.astype(np.float32), (H_new, W_new),
                                order=1, anti_aliasing=True,
                                preserve_range=True) > 0.5
            if gt.ndim == 2:
                gt = resize(gt, (H_new, W_new), order=1, anti_aliasing=True,
                            preserve_range=True).astype(np.float32)
            else:
                gt = resize(gt, (H_new, W_new, gt.shape[2]),
                            order=1, anti_aliasing=True,
                            preserve_range=True).astype(np.float32)
        else:
            uhsi, mask, amide1 = spatial_downsample(uhsi, mask, amide1, scale)
            depth_mask = ndimage_zoom(depth_mask.astype(np.float32),
                                      (scale, scale), order=0) > 0.5
            if gt.ndim == 2:
                gt = ndimage_zoom(gt, (scale, scale), order=1).astype(np.float32)
            else:
                gt = ndimage_zoom(gt, (scale, scale, 1), order=1).astype(np.float32)

        print(f"  After downsampling: x {uhsi.shape}, "
              f"y {gt.shape}, mask pixels: {mask.sum()}, "
              f"depth_mask pixels: {depth_mask.sum()}")

    # Refresh shape variables after possible downsampling
    H, W, C = uhsi.shape

    # ---- Check size is sufficient for requested patches ----
    margin = max(args.patch, args.out_patch) // 2
    if uhsi.shape[0] <= 2 * margin or uhsi.shape[1] <= 2 * margin:
        min_dim = 2 * margin + 1
        print(f"  WARNING: image size {uhsi.shape[:2]} too small for "
              f"patch={args.patch}/out_patch={args.out_patch} "
              f"(need >= {min_dim}). This sample will yield 0 patches.")


    # ---- Fill non-cartilage pixels in the input cube ----
    fill_mode = getattr(args, "nonmask_fill", "keep_zero")
    nonmask = ~mask
    n_nonmask = int(nonmask.sum())

    if fill_mode == "constant" and n_nonmask > 0:
        v = float(getattr(args, "nonmask_fill_value", 0.0))
        uhsi[nonmask, :] = v
        print(f"  Filled {n_nonmask:,} non-cartilage pixels with "
              f"constant={v} (mode=constant). After cartilage-only "
              f"x_scaler fit, this becomes a neutral background.")

    elif fill_mode == "zero" and n_nonmask > 0:
        uhsi[nonmask, :] = 0.0
        print(f"  Filled {n_nonmask:,} non-cartilage pixels with 0 "
              f"(mode=zero, explicit).")

    elif fill_mode == "patch_mean" and n_nonmask > 0:
        mflat = mask.reshape(-1).astype(bool)
        if mflat.any():
            mean_spec = (uhsi.reshape(-1, C)[mflat]
                         .mean(axis=0)
                         .astype(np.float32))
            uhsi[nonmask, :] = mean_spec
            print(f"  Filled {n_nonmask:,} non-cartilage pixels with the "
                  f"per-sample cartilage-mean spectrum (mode=patch_mean). "
                  f"NOTE: this is a sample-specific constant; the model can "
                  f"in principle pick up that constant near fold borders.")
        else:
            print("  nonmask_fill=patch_mean skipped (empty cartilage mask).")

    elif fill_mode == "keep_zero":
        # Leave the original behaviour. Fold pixels stay at whatever value
        # the fold-removal pipeline wrote (typically 0).
        pass


    # ---- Baseline correction ----
    uhsi_corr = apply_baseline_corr(
        uhsi, mask,
        mode=args.baseline_mode,
        emsc_degree=args.emsc_degree,
        wavelengths=wavelengths,
    )


    # ---- Extract patches from full image ----
    col_range = (0, W)  # use entire image

    X, y, m_out, m_in, coords = extract_samples(
        uhsi_corr, gt, mask, col_range,
        overlap=overlap_override if overlap_override is not None else args.overlap,
        in_patch=args.patch,
        in_avg_mode=args.avg_mode,
        out_patch=args.out_patch,
        out_avg_mode=args.out_avg_mode,
        min_in_mask_frac=args.min_in_mask_frac,
        min_out_mask_frac=args.min_out_mask_frac,
    )

    X, y, m_out, m_in, coords = filter_valid_with_min(X, y, m_out, m_in, coords)


    # Build auxiliary 2D maps (depth / pos / global) from the (possibly
    # downsampled) uHSI cube and cartilage mask, then crop one patch per
    # coordinate. The module-level extract_aux_patches() is used.
    aux_dict = build_aux_maps(uhsi, mask, args, depth_mask=depth_mask)
    has_any_aux = (len(aux_dict["full_maps"]) > 0
                   or len(aux_dict["patch_maps"]) > 0)
    X_aux = (extract_aux_patches(aux_dict, coords, args.patch)
             if has_any_aux else None)


    uhsi_shape = uhsi.shape
    # ---- FREE large arrays no longer needed ----
    del uhsi, uhsi_corr, amide1
    gc.collect()

    return X, y, m_out, m_in, coords, gt, mask, uhsi_shape, ftir_wls, X_aux

def load_mat_any(path):
    """Load a MAT-file regardless of version (<=7.2 or 7.3)."""
    try:
        return hdf5storage.loadmat(path)
    except Exception:
        return sio.loadmat(path, squeeze_me=True, struct_as_record=False)

def _get_field(obj, field):
    if isinstance(obj, dict):
        return obj[field]
    return getattr(obj, field)

def _has_field(obj, field):
    if isinstance(obj, dict):
        return field in obj
    return hasattr(obj, field)


def run_one_fold(args, samples, train_samples, val_samples, test_samples,
                fold_tag, results_base):
    """
    Run the full pipeline for ONE split (preprocess → train → evaluate → plot).
    Returns the metrics dict so the LOSO loop can aggregate them.
    """

    # ---- Per-fold results directory ----
    run_tag = make_run_tag(args)
    short_tag = make_short_tag(args, run_tag)

    # Build candidate path; if it exists, append _v2, _v3, ... to avoid overwrite
    candidate = os.path.join(results_base, short_tag, fold_tag)
    i = 1
    while True:
        results_dir = candidate if i == 1 else f"{candidate}_v{i}"
        try:
            os.makedirs(results_dir, exist_ok=False)
            break
        except FileExistsError:
            i += 1

    print(f"\n[{fold_tag}] Short tag: {short_tag}")
    print(f"[{fold_tag}] Full run_tag: {run_tag}")
    print(f"[{fold_tag}] Results: {results_dir}")

    # ---- BO trials skip heavy/redundant saves to keep disk + IO down ----
    _is_bo = bool(getattr(args, "_is_bo_trial", False))

    # ---- Per-trial wall-clock budget (cooperative) ----
    _fold_t0 = time.time()
    _trial_budget = int(getattr(args, "bo_trial_timeout_sec", 0) or 0)
    _section_times = {}

    def _section(label):
        """Context manager-like helper: record elapsed time, check budget."""
        class _Ctx:
            def __enter__(self_inner):
                self_inner.t = time.time()
                self_inner.label = label
                print(f"[time] >>> {label} (t={self_inner.t - _fold_t0:6.1f}s)",
                      flush=True)
                return self_inner
            def __exit__(self_inner, exc_type, exc, tb):
                dt = time.time() - self_inner.t
                _section_times[self_inner.label] = (
                    _section_times.get(self_inner.label, 0.0) + dt
                )
                print(f"[time] <<< {self_inner.label}: {dt:6.1f}s  "
                      f"(total {time.time() - _fold_t0:6.1f}s)", flush=True)
                if _is_bo and _trial_budget > 0 and \
                        (time.time() - _fold_t0) > _trial_budget:
                    print(f"[time] BUDGET EXCEEDED after '{self_inner.label}': "
                          f"{time.time() - _fold_t0:.1f}s > "
                          f"{_trial_budget}s — pruning trial.", flush=True)
                    raise optuna.TrialPruned()
        return _Ctx()

    def _check_budget(where):
        if _is_bo and _trial_budget > 0 and \
                (time.time() - _fold_t0) > _trial_budget:
            print(f"[time] BUDGET EXCEEDED at '{where}': "
                  f"{time.time() - _fold_t0:.1f}s > {_trial_budget}s — "
                  f"pruning trial.", flush=True)
            raise optuna.TrialPruned()

    # ---- Append to registry Excel (after results_dir is known) ----
    registry_path = os.path.join(results_base, REGISTRY_BASENAME)


    # Identifiers first
    if not _is_bo:
        row = {
            "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
            "short_tag":   short_tag,
            "fold_tag":    fold_tag,
            "results_dir": results_dir,
            "run_tag":     run_tag,
        }
        row.update(_args_to_excel_row(args))
        append_run_to_excel(registry_path, row)



   
    # ---- Tee logs to a per-fold log file ----
    _log_path = os.path.join(results_dir, "stdout.log")
    _log_fh = None
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    try:
        _log_fh = open(_log_path, "w", buffering=1, encoding="utf-8")
        sys.stdout = _Tee(_orig_stdout, _log_fh)
        sys.stderr = _Tee(_orig_stderr, _log_fh)

        # ---- Wavelengths ----
        wl_min = getattr(args, "uhsi_wl_min", 397.01)
        wl_max = getattr(args, "uhsi_wl_max", 1004.52)
        if args.patch_dir is not None:
            # Infer C_total from the first .npz file instead of raw data
            first_npz = None
            for fname in sorted(os.listdir(args.patch_dir)):
                if fname.endswith("_train.npz"):
                    first_npz = os.path.join(args.patch_dir, fname)
                    break
            if first_npz is None:
                raise FileNotFoundError(
                    f"No _train.npz files in {args.patch_dir}"
                )
            with np.load(first_npz, allow_pickle=True) as tmp:
                C_total = tmp["X"].shape[-1]
            wls = np.linspace(wl_min, wl_max, C_total).astype(np.float32)
            if getattr(args, "exclude_wl_ranges", None):
                print("[wl-exclude] WARNING: --exclude_wl_ranges is ignored "
                      "in --patch_dir mode; exclusion must be applied at "
                      "patch extraction time.")
        else:
            # Load one raw file to get RAW C (before any exclusion)
            with h5py.File(samples[0]["uhsi_path"], "r") as f:
                C_total_raw = f[args.uhsi_key].shape[0]
            # Always pass the RAW wavelength axis to workers. Per-sample
            # exclusion happens inside load_and_extract_one_sample.
            wls = np.linspace(wl_min, wl_max, C_total_raw).astype(np.float32)

            # For placeholder shape calculations, we need to know the
            # POST-exclusion channel count. Compute it once here without
            # mutating `wls`.
            if getattr(args, "exclude_wl_ranges", None):
                _, wls_trimmed, _ = apply_wavelength_exclusion(
                    np.zeros((1, 1, C_total_raw), dtype=np.float32),
                    wls, args.exclude_wl_ranges,
                )
                C_total = int(wls_trimmed.shape[0])
                print(f"[wl-exclude] Raw uHSI channels: {C_total_raw}; "
                      f"after excluding {args.exclude_wl_ranges}: "
                      f"{C_total} channels.")
                del wls_trimmed
            else:
                C_total = C_total_raw


        if args.patch_dir is not None:

            # ---- FAST PATH: load pre-extracted patches ----
            if args.target_mode == "spectra":
                raise NotImplementedError(
                    "--patch_dir is only supported for target_mode=map. "
                    "Run without --patch_dir for spectra mode."
                )
            print(f"Loading pre-extracted patches from: {args.patch_dir}\n")

            # Load config from patch dir for verification
            config_path = os.path.join(args.patch_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    patch_config = json.load(f)
                print(f"Patch config: patch={patch_config.get('patch')}, "
                    f"out_patch={patch_config.get('out_patch')}, "
                    f"baseline={patch_config.get('baseline_mode')}, "
                    f"smooth={patch_config.get('smooth_x')}")
                # Verify consistency
                if patch_config.get("patch") != args.patch:
                    print(f"WARNING: patch size mismatch: "
                        f"patches={patch_config.get('patch')}, args={args.patch}")
                if patch_config.get("out_patch") != args.out_patch:
                    print(f"WARNING: out_patch mismatch: "
                        f"patches={patch_config.get('out_patch')}, args={args.out_patch}")
                print()
            # Decide which map(s) to load
            if args.model == "multi_unet" and args.map_fields:
                mf_arg = list(args.map_fields)
            else:
                mf_arg = args.map_field

            X_train, y_train, m_train, coords_train, meta_train, X_aux_train = \
                load_patches_from_npz_parallel(
                    args.patch_dir, train_samples, "train", args, wls,
                    use_test_patches=False, map_field=mf_arg)
            X_val, y_val, m_val, coords_val, meta_val, X_aux_val = \
                load_patches_from_npz_parallel(
                    args.patch_dir, val_samples, "val", args, wls,
                    use_test_patches=False, map_field=mf_arg)
            X_test, y_test, m_test, coords_test, meta_test, X_aux_test = \
                load_patches_from_npz_parallel(
                    args.patch_dir, test_samples, "test", args, wls,
                    use_test_patches=True, map_field=mf_arg)


            # The fast path does not store m_in. Consistent x-scaler
            # fitting requires the input cartilage mask per patch;
            # falling back to all-ones would bias the scaler statistics
            # by including fold-zeroed pixels. The pipeline therefore
            # aborts in that case.

            print("[fast path] WARNING: m_in is not stored in "
                  "pre-extracted patches.")
            if X_train.ndim == 4 and m_train.shape[1:3] == (args.patch, args.patch):
                # The output mask happens to have the same spatial size
                # as the input patch. This is correct only when
                # in_patch == out_patch. We accept it as a substitute
                # for m_in to preserve the publication-grade protocol.
                m_in_train = m_train.copy()
                m_in_val   = m_val.copy()   if X_val.size > 0 else np.empty((0, args.patch, args.patch), dtype=bool)
                m_in_test  = m_test.copy()  if X_test.size > 0 else np.empty((0, args.patch, args.patch), dtype=bool)
                print("[fast path] Using m_out as m_in substitute "
                      "(in_patch == out_patch).")
            elif X_train.ndim == 4:
                raise RuntimeError(
                    "[fast path] in_patch != out_patch and m_in was not "
                    "saved with the patches. The x-scaler would be fit "
                    "on all pixels (including fold-zeroed ones), which "
                    "violates the fit-on-cartilage-only protocol. "
                    "Either (a) save m_in at patch extraction time, "
                    "or (b) re-extract with in_patch == out_patch."
                )

            else:
                m_in_train = np.ones(X_train.shape[0], dtype=bool)
                m_in_val   = np.ones(X_val.shape[0],   dtype=bool)
                m_in_test  = np.ones(X_test.shape[0],  dtype=bool)



        else:
            # ---- Load and extract patches per split ----
            def load_split(sample_list, split_name, overlap_override=None):
                """Load all samples in a split, concatenate patches."""
                X_all, y_all, m_all, coords_all = [], [], [], []
                aux_all = []
                meta_list = []

                m_in_all = []
                for s in sample_list:
                    print(f"Loading {split_name}: {s['name']} ...")
                    X, y, m_out, m_in, coords, gt, mask_s, shape, ftir_wls, X_aux = \
                        load_and_extract_one_sample(
                            s["ftir_path"], s["uhsi_path"], args,
                            wavelengths=wls,
                            overlap_override=overlap_override,
                            ftir_cube_path=s.get("ftir_cube_path"),
                        )
                    X_all.append(X)
                    y_all.append(y)
                    m_all.append(m_out)
                    m_in_all.append(m_in)
                    coords_all.extend(coords)
                    # Per-sample aux: if disabled, store a zero-channel placeholder
                    # so concatenation along the channel axis stays consistent.
                    if X_aux is None:
                        aux_all.append(
                            np.zeros((X.shape[0], args.patch, args.patch, 0),
                                     dtype=np.float32)
                        )
                    else:
                        aux_all.append(X_aux.astype(np.float32))
                    meta_list.append({
                        "name": s["name"],
                        "map_data": gt,           # 2D in map mode, 3D cube in spectra mode
                        "mask": mask_s,
                        "shape": shape,
                        "n_patches": X.shape[0],
                        "coords": coords,
                        "ftir_wls": ftir_wls,
                        "ftir_cube_path": s.get("ftir_cube_path"),
                    })
                if X_all:
                    total_elems = sum(x.size for x in X_all)
                    total_gb = total_elems * 4 / (1024**3)
                    print(f"{split_name}: estimated X memory before concat = {total_gb:.2f} GiB")

                if X_all:
                    X_cat = np.concatenate(X_all, axis=0)
                    del X_all
                    y_cat = np.concatenate(y_all, axis=0)
                    del y_all
                    m_cat = np.concatenate(m_all, axis=0)
                    del m_all
                    m_in_cat = np.concatenate(m_in_all, axis=0)
                    del m_in_all
                    X_aux_cat = np.concatenate(aux_all, axis=0)
                    del aux_all
                    gc.collect()

                else:
                    # ---- Empty split: build empty placeholders ----
                    if args.avg_mode == "patch":
                        X_cat = np.empty((0, args.patch, args.patch, C_total),
                                         dtype=np.float32)
                    else:
                        X_cat = np.empty((0, C_total), dtype=np.float32)

                    if args.target_mode == "map":
                        if args.out_avg_mode == "patch":
                            y_cat = np.empty((0, args.out_patch, args.out_patch),
                                             dtype=np.float32)
                        elif args.out_avg_mode == "flat":
                            y_cat = np.empty((0, args.out_patch * args.out_patch),
                                             dtype=np.float32)
                        else:  # avg
                            y_cat = np.empty((0,), dtype=np.float32)
                    else:  # spectra mode
                        # Use the correct spectral channel count (623) so any
                        # downstream concatenation does not fail on dim mismatch.
                        C_ftir_est = int(FTIR_WLS_OVERRIDE.size)
                        if args.out_avg_mode == "patch":
                            y_cat = np.empty(
                                (0, args.out_patch, args.out_patch, C_ftir_est),
                                dtype=np.float32)
                        elif args.out_avg_mode == "flat":
                            y_cat = np.empty(
                                (0, args.out_patch * args.out_patch * C_ftir_est),
                                dtype=np.float32)
                        else:  # avg
                            y_cat = np.empty((0, C_ftir_est), dtype=np.float32)

                    m_cat = np.empty((0, args.out_patch, args.out_patch),
                                     dtype=np.bool_)
                    # Empty aux placeholder: 0 channels, matches input patch grid
                    X_aux_cat = np.empty((0, args.patch, args.patch, 0),
                                         dtype=np.float32)
                    m_in_cat = np.empty((0, args.patch, args.patch), dtype=np.bool_)

                return X_cat, y_cat, m_cat, m_in_cat, coords_all, meta_list, X_aux_cat


            with _section("load_split_parallel_mat[train]"):
                X_train, y_train, m_train, m_in_train, coords_train, meta_train, X_aux_train = \
                    load_split_parallel_mat(
                        train_samples, "train", args, wls,
                        overlap_override=None, C_total=C_total,
                    )
            with _section("load_split_parallel_mat[val]"):
                X_val, y_val, m_val, m_in_val, coords_val, meta_val, X_aux_val = \
                    load_split_parallel_mat(
                        val_samples, "val", args, wls,
                        overlap_override=None, C_total=C_total,
                    )
            P_ref = max(args.patch, args.out_patch)
            test_overlap = max(args.overlap, 1.0 - args.out_patch / P_ref)
            print(f"Test overlap: {test_overlap:.4f} "
                  f"(step ≈ {max(1, int(round(P_ref * (1.0 - test_overlap))))})")
            with _section("load_split_parallel_mat[test]"):
                X_test, y_test, m_test, m_in_test, coords_test, meta_test, X_aux_test = \
                    load_split_parallel_mat(
                        test_samples, "test", args, wls,
                        overlap_override=test_overlap, C_total=C_total,
                    )



        print(f"\nAfter loading:")
        print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
        print(f"  X_val:   {X_val.shape}    y_val:   {y_val.shape}")
        print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}\n")

        gc.collect()

        # ---- STREAMING PREPROCESSING SWITCH ----
        # When True, SG / spectrum_norm / derivatives / x_scaler.transform
        # are deferred to the DataLoader workers (UHsiStreamingDataset).
        # This reduces peak memory dramatically and overlaps CPU
        # preprocessing with GPU training.
        # Enabled for patch-input UNet/CNN models in BOTH target_mode=map
        # AND target_mode=spectra (the X-side transforms are per-spectrum
        # / per-patch in either case, so the streaming path is bit-exact
        # to the eager path).
        # NOTE: this also applies to gan_mode != 'none', because the GAN
        # generator IS one of these models (UNet / MultiMapUNet).
        _use_streaming = bool(getattr(args, "use_streaming", False)) and (
            args.avg_mode == "patch"
            and args.model in ("unet", "unet_simple", "pretrained_unet",
                            "multi_unet", "cnn")
        )


        # Both the .npz worker and the .mat worker DEFER SG / SNV /
        # derivatives to the streaming dataset (via _defer_to_stream=True
        # inside each worker). So the streaming dataset must perform them
        # itself, with the user's chosen settings.
        # If, in the future, _defer_to_stream is set to False in the
        # workers, set _preproc_done_in_workers to True here and switch
        # the streaming flags below to (False, "none", "raw"). Today the
        # contract is: workers DO NOT touch X spectrally; the stream does.
        _preproc_done_in_workers = False
        _stream_smooth_x = args.smooth_x
        _stream_spectrum_norm = args.spectrum_norm
        _stream_der_mode = args.der_mode

        if _use_streaming:
            print("[streaming] Deferring SG / spectrum_norm / derivatives / "
                  "x_scaler.transform to DataLoader workers.")


        # ---- Single-pass Savitzky–Golay (smoothing + derivatives in one fit) ---- #
        # Skipped when streaming: the DataLoader workers do this per patch.
        # This block is mathematically equivalent to the streaming
        # UHsiStreamingDataset._preprocess_one Step 1: ONE polynomial fit yields
        # the smoothed signal (deriv=0) and its derivatives (deriv=1, deriv=2),
        # with consistent window length and polynomial order.
        if (not _preproc_done_in_workers and not _use_streaming
                and (args.smooth_x or args.der_mode != "raw")):

            print(f"Applying single-pass Savitzky–Golay "
                f"(smooth_x={args.smooth_x}, der_mode={args.der_mode}, "
                f"window={args.sg_window}, poly={args.sg_poly})\n")

            def _eager_sg_block(X, sg_window, sg_poly, smooth_x, der_mode,
                                delta_wl):
                """
                Apply the SAME single-pass SG pipeline that
                UHsiStreamingDataset._preprocess_one uses, to a full array X
                with shape (N, C) or (N, P, P, C).
                """
                if X is None or X.size == 0:
                    return X

                C = X.shape[-1]
                if C < 2:
                    return X.astype(np.float32)

                # Force odd window and clip to channel count
                w = sg_window
                if w % 2 == 0:
                    w += 1
                w = min(w, C // 2 * 2 + 1)
                p = min(sg_poly, w - 1)

                needs_smooth = bool(smooth_x)
                needs_der1 = der_mode in ("der1", "raw+der1",
                                        "raw+der1+der2", "der1+der2")
                needs_der2 = der_mode in ("der2", "raw+der2",
                                        "raw+der1+der2", "der1+der2")
                keeps_raw = der_mode in ("raw", "raw+der1", "raw+der2",
                                        "raw+der1+der2")

                # Nothing to do — should be unreachable because the outer
                # guard already filtered out (not smooth_x and der_mode == raw).
                if not (needs_smooth or needs_der1 or needs_der2):
                    return X.astype(np.float32)

                if X.ndim == 4:
                    N, P1, P2, _ = X.shape
                    X_flat = X.reshape(-1, C)
                elif X.ndim == 2:
                    N = X.shape[0]
                    P1 = P2 = None
                    X_flat = X
                else:
                    raise ValueError(f"_eager_sg_block: unsupported X.ndim={X.ndim}")

                parts = []
                if keeps_raw:
                    if needs_smooth:
                        base = savgol_filter(X_flat, window_length=w, polyorder=p,
                                            deriv=0, axis=-1).astype(np.float32)
                    else:
                        base = X_flat.astype(np.float32)
                    parts.append(base)

                if needs_der1:
                    d1 = savgol_filter(X_flat, window_length=w, polyorder=p,
                                    deriv=1, delta=delta_wl,
                                    axis=-1).astype(np.float32)
                    parts.append(d1)

                if needs_der2:
                    d2 = savgol_filter(X_flat, window_length=w, polyorder=p,
                                    deriv=2, delta=delta_wl,
                                    axis=-1).astype(np.float32)
                    parts.append(d2)

                Xc = np.concatenate(parts, axis=-1)
                if P1 is not None:
                    Xc = Xc.reshape(N, P1, P2, -1)
                return Xc.astype(np.float32)

            delta = float(wls[1] - wls[0]) if len(wls) > 1 else 1.0

            X_train = _eager_sg_block(
                X_train, args.sg_window, args.sg_poly,
                args.smooth_x, args.der_mode, delta,
            )
            gc.collect()
            if X_val.size > 0:
                X_val = _eager_sg_block(
                    X_val, args.sg_window, args.sg_poly,
                    args.smooth_x, args.der_mode, delta,
                )
            if X_test.size > 0:
                X_test = _eager_sg_block(
                    X_test, args.sg_window, args.sg_poly,
                    args.smooth_x, args.der_mode, delta,
                )
            gc.collect()

        # ---- Per-spectrum / per-patch normalization (no leakage) ---- #
        # Applied AFTER the single-pass SG block above, matching the
        # streaming dataset's ordering: SG → spectrum_norm → x_scaler → DR.
        # Skipped when streaming: the DataLoader workers apply this per patch.
        if args.spectrum_norm != "none" and not _preproc_done_in_workers and not _use_streaming:
            print(f"Applying per-spectrum normalization: {args.spectrum_norm}\n")
            X_train = apply_spectrum_norm(X_train, mode=args.spectrum_norm)
            if X_val.size > 0:
                X_val = apply_spectrum_norm(X_val, mode=args.spectrum_norm)
            if X_test.size > 0:
                X_test = apply_spectrum_norm(X_test, mode=args.spectrum_norm)
            gc.collect()


        # ---- Output (y) smoothing and derivatives for spectra-mode targets ---- #
        # Unchanged: y-side preprocessing uses simple backward differences
        # (apply_derivatives_simple) on the FTIR wavenumber axis, which is
        # physically the correct operator for FTIR spectra. Optional Savitzky–Golay
        # smoothing of y is still a single-pass SG with deriv=0 (no derivative
        # stacking on the y side), so the existing implementation is correct.
        if args.target_mode == "spectra":
            if args.smooth_y:
                print(f"Smoothing outputs with Savitzky-Golay (target_mode=spectra)\n")
                y_train = apply_savgol_smoothing(y_train, window=args.sg_window, polyorder=args.sg_poly)
                if y_val.size > 0:
                    y_val = apply_savgol_smoothing(y_val, window=args.sg_window, polyorder=args.sg_poly)
                if y_test.size > 0:
                    y_test = apply_savgol_smoothing(y_test, window=args.sg_window, polyorder=args.sg_poly)

            if args.out_der_mode != "raw":
                print(f"Applying output derivatives: {args.out_der_mode}\n")
                y_train = apply_derivatives_simple(y_train, der_mode=args.out_der_mode, ftir_wls=meta_train[0]["ftir_wls"])
                if y_val.size > 0:
                    y_val = apply_derivatives_simple(y_val, der_mode=args.out_der_mode, ftir_wls=meta_val[0]["ftir_wls"])
                if y_test.size > 0:
                    y_test = apply_derivatives_simple(y_test, der_mode=args.out_der_mode, ftir_wls=meta_test[0]["ftir_wls"])




        # ---- DIAGNOSTIC + sanitize X for NaN/Inf (memory-safe, chunked) ----
        def _count_nonfinite_chunked(arr, name):
            """Count NaN/Inf in chunks to avoid allocating full-size bool masks."""
            if arr is None or arr.size == 0:
                return
            chunk_rows = max(1, 200_000_000 // max(1, arr[0].nbytes))
            n_nan = 0
            n_inf = 0
            for i in range(0, arr.shape[0], chunk_rows):
                sl = arr[i:i + chunk_rows]
                n_nan += int(np.isnan(sl).sum())
                n_inf += int(np.isinf(sl).sum())
            print(f"  {name}: NaN={n_nan:,}  Inf={n_inf:,}")

        _count_nonfinite_chunked(X_train, "X_train")
        _count_nonfinite_chunked(X_val,   "X_val")
        _count_nonfinite_chunked(X_test,  "X_test")

        X_train = sanitize_inplace(X_train, "X_train")
        if X_val.size > 0:
            X_val = sanitize_inplace(X_val, "X_val")
        if X_test.size > 0:
            X_test = sanitize_inplace(X_test, "X_test")

        gc.collect()


        Xflat = X_train.reshape(-1, X_train.shape[-1])
        stds = np.std(Xflat, axis=0)
        print("Channels with std < 1e-10:", np.sum(stds < 1e-10))
        print("Channels with std < 1e-6:", np.sum(stds < 1e-6))
        print("Min std:", stds.min(), "at channel", np.argmin(stds))

        del Xflat, stds
        gc.collect()

        
        # ---- Scaling of inputs (X) with sklearn ---- #
        x_sklearn = None

        if args.x_scaler != "none":
            print(f"Scaling inputs with sklearn {args.x_scaler} scaler...\n")
            x_sklearn = make_sklearn_scaler(args.x_scaler)

            # Build a flat cartilage mask aligned to (N*P*P) to fit the scaler
            # ONLY on cartilage pixels.
            if X_train.ndim == 2:
                Xtr_view = X_train
                tr_shape = X_train.shape
                if m_in_train is not None and m_in_train.ndim == 1 \
                        and m_in_train.shape[0] == X_train.shape[0]:
                    m_fit_full = m_in_train.astype(bool)
                else:
                    m_fit_full = np.ones(Xtr_view.shape[0], dtype=bool)
            elif X_train.ndim == 4:
                N, P1, P2, C = X_train.shape
                Xtr_view = X_train.reshape(-1, C)   # view, not a copy
                tr_shape = (N, P1, P2, C)
                if (m_in_train is not None and m_in_train.ndim == 3
                        and m_in_train.shape[1:] == tr_shape[1:3]):
                    m_fit_full = m_in_train.reshape(-1).astype(bool)
                else:
                    print("  x_scaler: m_in_train shape mismatch; "
                          "falling back to fitting on all pixels.")
                    m_fit_full = np.ones(Xtr_view.shape[0], dtype=bool)
            else:
                raise ValueError(
                    f"Unsupported X_train.ndim={X_train.ndim} for scaling"
                )

            n_fit_total = Xtr_view.shape[0]
            n_fit_used  = int(m_fit_full.sum())

            # Subsample for fit when streaming, to keep peak memory low.
            # Fitting a Standard/Robust/Quantile scaler on 2M cartilage
            # pixels is statistically indistinguishable from fitting on
            # tens of millions; this is standard for large HSI datasets.
            # Streaming branch in run_one_fold, before fitting x_sklearn:
            if _use_streaming and n_fit_used > 2_000_000:
                rng = np.random.default_rng(args.seed)
                cand_idx = np.where(m_fit_full)[0]
                sub_idx = rng.choice(cand_idx, size=2_000_000, replace=False)
                # IMPORTANT: do NOT materialise X_train.reshape(-1, C) as a copy.
                # numpy.reshape on a contiguous array returns a view; indexing into
                # a view allocates only the 2e6 × C subsample, not the whole tensor.
                X_for_fit = Xtr_view[sub_idx]
                print(f"  x_scaler fit (streaming) on subsample of "
                      f"{X_for_fit.shape[0]:,} / {n_fit_total:,} cartilage "
                      f"pixels (seed={args.seed}).")
            elif n_fit_used >= 100:
                X_for_fit = Xtr_view[m_fit_full]
                print(f"  x_scaler fit on {n_fit_used:,} / {n_fit_total:,} "
                      f"cartilage pixels "
                      f"({100*n_fit_used/max(n_fit_total,1):.1f}%).")
            else:
                X_for_fit = Xtr_view
                print(f"  WARNING: too few cartilage pixels "
                      f"({n_fit_used}); fitting x_scaler on all pixels.")
                            
            # ---- Chemometrically correct ordering for the x-scaler fit ----
            # The streaming DataLoader worker applies, per patch, the sequence
            #   SG smoothing (optional) -> SG derivatives -> spectrum_norm
            #   -> x_scaler.transform -> DR
            # (Rinnan, van den Berg & Engelsen, TrAC 2009).
            # The scaler MUST be fit on data that has gone through the same
            # chain up to and including spectrum_norm; otherwise the feature
            # count and distribution seen at fit and at transform disagree.
            #
            # This block builds a fit slice that reproduces the worker's
            # pipeline EXACTLY, in the same order:
            #   (1) SG smoothing on the raw cartilage spectra
            #   (2) SG derivatives (may multiply channel count by 2 or 3)
            #   (3) Per-spectrum normalization (may double channel count
            #       for 'snv+raw' / 'l2+raw')
            # After these three steps, X_for_fit has the SAME channel count
            # that the worker will feed x_sklearn.transform().
            #
            # Steps 1-2 are already applied eagerly to X_train in the non-
            # streaming path, but the x_scaler is fit BEFORE spectrum_norm
            # is applied to the fit slice. Under streaming, none of these
            # three steps has been applied yet. In either case, the fix is
            # to run the missing steps on X_for_fit only.
            #
            # We take advantage of the fact that X_for_fit is a small
            # subsample (<= 2M rows) so re-applying these operators is cheap.

            delta = float(wls[1] - wls[0]) if len(wls) > 1 else 1.0

            if _use_streaming:
                # Under streaming, X_for_fit is a slice of the RAW cube.
                # Apply the full pre-scaler chain in canonical order.
                if _stream_smooth_x:
                    X_for_fit = apply_savgol_smoothing(
                        X_for_fit,
                        window=args.sg_window,
                        polyorder=args.sg_poly,
                    )
                    print(f"  x_scaler fit slice: applied SG smoothing "
                          f"(w={args.sg_window}, p={args.sg_poly}).")

                if _stream_der_mode != "raw":
                    X_for_fit = apply_derivatives(
                        X_for_fit,
                        der_mode=_stream_der_mode,
                        sg_window=args.sg_window,
                        polyorder=args.sg_poly,
                        delta=delta,
                    )
                    print(f"  x_scaler fit slice: applied "
                          f"{_stream_der_mode} derivatives; channel "
                          f"count now {X_for_fit.shape[-1]}.")

                if _stream_spectrum_norm != "none":
                    X_for_fit = apply_spectrum_norm(
                        X_for_fit, mode=_stream_spectrum_norm,
                    )
                    print(f"  x_scaler fit slice: applied "
                          f"spectrum_norm={_stream_spectrum_norm}; "
                          f"channel count now {X_for_fit.shape[-1]}.")
            else:
                # Eager path: SG smoothing and derivatives have already
                # been applied to the full X_train (and therefore to
                # X_for_fit, which is a view of X_train). But
                # spectrum_norm has ALSO already been applied to X_train
                # in the non-streaming branch above, so nothing to redo
                # here. The channel count of X_for_fit already matches
                # what x_sklearn.transform() will see at inference.
                pass

            # Sanitize fit data: drop non-finite rows and clip extreme
            # percentiles so PowerTransformer / QuantileTransformer
            # cannot fail with BracketError / overflow on degenerate
            # channels.
            finite_rows = np.isfinite(X_for_fit).all(axis=1)

            n_dropped = int((~finite_rows).sum())
            if n_dropped > 0:
                print(f"  x_scaler: dropping {n_dropped:,} non-finite "
                      f"rows before fit.")
            X_for_fit = X_for_fit[finite_rows]
            if X_for_fit.shape[0] < 100:
                print(f"  x_scaler: too few finite rows "
                      f"({X_for_fit.shape[0]}); pruning trial.")
                raise optuna.TrialPruned()

            lo = np.nanpercentile(X_for_fit, 0.1, axis=0)
            hi = np.nanpercentile(X_for_fit, 99.9, axis=0)
            # Guard against constant channels (lo == hi)
            eq = np.isclose(lo, hi)
            if eq.any():
                hi = np.where(eq, lo + 1e-6, hi)
            X_for_fit = np.clip(X_for_fit, lo, hi).astype(np.float64)

            try:
                x_sklearn.fit(X_for_fit)
            except Exception as e:
                print(f"  x_scaler.fit failed "
                      f"({type(e).__name__}: {e}); pruning trial.")
                del X_for_fit
                gc.collect()
                raise optuna.TrialPruned()

            # ---- Consistency invariant check ----
            # Assert that the scaler expects the same channel count
            # that the streaming worker will pass at inference time.
            # A mismatch indicates that the ordering of SG /
            # spectrum_norm / x_scaler in the fit path has drifted
            # from the ordering in UHsiStreamingDataset._preprocess_one,
            # which would produce corrupted predictions. The pipeline
            # aborts here rather than proceed with an inconsistent
            # feature pipeline.

            n_features_fit = int(X_for_fit.shape[-1])
            n_features_expected = getattr(x_sklearn, "n_features_in_", None)
            if (n_features_expected is not None
                    and n_features_expected != n_features_fit):
                raise RuntimeError(
                    f"[x_scaler] invariant violation: scaler was fit on "
                    f"{n_features_fit} features but reports "
                    f"n_features_in_={n_features_expected}. "
                    f"Refusing to proceed."
                )
            print(f"  x_scaler fitted on {n_features_fit} features "
                  f"(matches streaming worker channel count).")

            del X_for_fit
            gc.collect()



            if not _use_streaming:
                # ---- Eager transform path (chunked for memory safety) ----
                def _safe_transform(scaler, X_flat, chunk_rows=500_000):
                    """Apply scaler.transform in chunks and clip to safe range.
                    Returns sanitized float32 array. Raises TrialPruned
                    on failure during BO."""
                    try:
                        out = np.empty_like(X_flat, dtype=np.float32)
                        for i in range(0, X_flat.shape[0], chunk_rows):
                            end = min(i + chunk_rows, X_flat.shape[0])
                            Z = scaler.transform(X_flat[i:end])
                            Z = np.clip(Z, -1e6, 1e6)
                            Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
                            out[i:end] = Z.astype(np.float32)
                        return out
                    except Exception as e:
                        print(f"  x_scaler.transform failed "
                              f"({type(e).__name__}: {e}); pruning.")
                        raise optuna.TrialPruned()

                X_train = _safe_transform(x_sklearn, Xtr_view) \
                    .reshape(tr_shape)
                del Xtr_view
                gc.collect()

                print("X_train finite:", np.isfinite(X_train).all())
                print("X_train min/max:",
                      np.nanmin(X_train), np.nanmax(X_train))
                print("X_train std per channel:",
                      np.std(X_train.reshape(-1, X_train.shape[-1]),
                             axis=0)[:5])

                # val
                if X_val.size > 0:
                    if X_val.ndim == 2:
                        X_val = _safe_transform(x_sklearn, X_val)
                    else:
                        Nv, P1v, P2v, Cv = X_val.shape
                        X_val = _safe_transform(
                            x_sklearn, X_val.reshape(-1, Cv)
                        ).reshape(Nv, P1v, P2v, Cv)

                # test
                if X_test.size > 0:
                    if X_test.ndim == 2:
                        X_test = _safe_transform(x_sklearn, X_test)
                    else:
                        Nt, P1t, P2t, Ct = X_test.shape
                        X_test = _safe_transform(
                            x_sklearn, X_test.reshape(-1, Ct)
                        ).reshape(Nt, P1t, P2t, Ct)

            else:
                # Streaming: do NOT transform here. Workers will apply
                # x_sklearn.transform() to each patch inside __getitem__.
                print("  [streaming] Skipping eager x_scaler.transform; "
                      "will run inside DataLoader workers.")
        else:
            print("Skipping input scaling (--x_scaler none)\n")

        # ---- Dimensionality Reduction ---- #
        # Propagate args.seed into apply_dim_reduction's PLSR-DR pixel
        # subsampling (and any future RNG inside the function), so the
        # DR fit is reproducible from a single seed.
        apply_dim_reduction._seed = int(args.seed)

        if args.dr_mode != "none":
            print(f"Applying dimensionality reduction: {args.dr_mode}, "
                f"n_components={args.dr_components}\n")
            if not _use_streaming:
                # Eager path: X has already been x-scaled; fit AND
                # transform on full arrays.
                X_train, X_val, X_test, dr_model = apply_dim_reduction(
                    X_train, X_val, X_test,
                    mode=args.dr_mode,
                    n_components=args.dr_components,
                    y_train=y_train,
                    m_train=m_train,
                    m_in_train=m_in_train,
                )
                print(f"After dimensionality reduction: "
                      f"X_train shape={X_train.shape} "
                      f"X_val shape={X_val.shape} "
                      f"X_test shape={X_test.shape}\n")
            else:
                # Streaming path: fit DR on a subsample, but do NOT
                # transform X_train/X_val/X_test here. The streaming
                # dataset will call dr_model.transform(...) per patch.
                #
                # The streaming dataset applies, in order:
                #   SG -> SNV -> derivatives -> x_scaler -> dr_model
                # dr_model must therefore be fit on data that has had
                # the same transformations applied. Otherwise
                # dr_model.transform() would receive a different
                # feature distribution than it was fit on.
                print("[streaming] Fitting dr_model on a subsample; "
                      "transform deferred to DataLoader workers.")
                if X_train.ndim == 4:
                    N_tr = X_train.shape[0]
                    n_keep = min(N_tr, 1024)
                    if n_keep < N_tr:
                        rng_dr = np.random.default_rng(args.seed + 1)
                        idx_dr = rng_dr.choice(N_tr, size=n_keep,
                                               replace=False)
                        X_slice = X_train[idx_dr].copy()
                        m_slice = (m_train[idx_dr]
                                   if m_train is not None else None)
                        min_slice = (m_in_train[idx_dr]
                                     if m_in_train is not None else None)
                        if y_train.ndim == 1:
                            y_slice = y_train[idx_dr]
                        else:
                            y_slice = y_train[idx_dr]
                    else:
                        X_slice = X_train.copy()
                        m_slice = m_train
                        min_slice = m_in_train
                        y_slice = y_train
                else:
                    X_slice = X_train.copy()
                    m_slice = m_train
                    min_slice = m_in_train
                    y_slice = y_train

                # Apply the SAME single-pass SG + spectrum_norm pipeline that the
                # streaming dataset will apply per patch, so dr_model sees the
                # right feature distribution.
                needs_smooth = bool(_stream_smooth_x)
                needs_der1 = _stream_der_mode in ("der1", "raw+der1",
                                                "raw+der1+der2", "der1+der2")
                needs_der2 = _stream_der_mode in ("der2", "raw+der2",
                                                "raw+der1+der2", "der1+der2")
                keeps_raw = _stream_der_mode in ("raw", "raw+der1", "raw+der2",
                                                "raw+der1+der2")

                delta = float(wls[1] - wls[0]) if len(wls) > 1 else 1.0

                if needs_smooth or needs_der1 or needs_der2:
                    C_slice = X_slice.shape[-1]
                    w = args.sg_window
                    if w % 2 == 0:
                        w += 1
                    w = min(w, C_slice // 2 * 2 + 1)
                    p = min(args.sg_poly, w - 1)

                    if X_slice.ndim == 4:
                        N_s, P1_s, P2_s, _ = X_slice.shape
                        X_flat_slice = X_slice.reshape(-1, C_slice)
                    else:
                        N_s = X_slice.shape[0]
                        X_flat_slice = X_slice

                    parts = []
                    if keeps_raw:
                        if needs_smooth:
                            base = savgol_filter(
                                X_flat_slice, window_length=w, polyorder=p,
                                deriv=0, axis=-1).astype(np.float32)
                        else:
                            base = X_flat_slice.astype(np.float32)
                        parts.append(base)

                    if needs_der1:
                        d1 = savgol_filter(
                            X_flat_slice, window_length=w, polyorder=p,
                            deriv=1, delta=delta, axis=-1).astype(np.float32)
                        parts.append(d1)

                    if needs_der2:
                        d2 = savgol_filter(
                            X_flat_slice, window_length=w, polyorder=p,
                            deriv=2, delta=delta, axis=-1).astype(np.float32)
                        parts.append(d2)

                    X_slice = np.concatenate(parts, axis=-1)
                    if X_slice.ndim != 4 and 'P1_s' in dir():
                        X_slice = X_slice.reshape(N_s, P1_s, P2_s, -1)
                    elif X_slice.ndim != 4:
                        pass  # 2D input stays 2D

                if _stream_spectrum_norm != "none":
                    X_slice = apply_spectrum_norm(
                        X_slice, mode=_stream_spectrum_norm,
                    )
                if x_sklearn is not None:
                    shp = X_slice.shape
                    try:
                        Z = x_sklearn.transform(
                            X_slice.reshape(-1, shp[-1])
                        )
                    except Exception as e:
                        print(f"[streaming] x_scaler.transform on dr_model "
                              f"fit slice failed "
                              f"({type(e).__name__}: {e}); pruning trial.")
                        raise optuna.TrialPruned()
                    Z = np.clip(Z, -1e6, 1e6)
                    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
                    X_slice = Z.reshape(shp).astype(np.float32)


                _, _, _, dr_model = apply_dim_reduction(
                    X_slice, None, None,
                    mode=args.dr_mode,
                    n_components=args.dr_components,
                    y_train=y_slice,
                    m_train=m_slice,
                    m_in_train=min_slice,
                )
                del X_slice, m_slice, min_slice, y_slice
                gc.collect()
                print(f"[streaming] dr_model fitted "
                      f"({type(dr_model).__name__}); X arrays unchanged.")
        else:
            dr_model = None


        # ---- Append auxiliary channels (depth / pos / global) ---- #
        # These must NOT pass through spectral preprocessing or x_scaler.
        # They are concatenated on the last axis just before y-scaling.
        def _append_aux(X, X_aux, name=""):
            if X_aux is None or X_aux.size == 0 or X_aux.shape[-1] == 0:
                return X
            if X.ndim != 4:
                raise ValueError(
                    f"Aux channels require avg_mode=patch (4D X); "
                    f"got X.ndim={X.ndim} for {name}"
                )
            if X.shape[0] != X_aux.shape[0]:
                raise ValueError(
                    f"Aux/X N mismatch for {name}: "
                    f"X={X.shape[0]} aux={X_aux.shape[0]}"
                )
            if X.shape[1:3] != X_aux.shape[1:3]:
                raise ValueError(
                    f"Aux/X spatial mismatch for {name}: "
                    f"X={X.shape[1:3]} aux={X_aux.shape[1:3]}"
                )
            return np.concatenate([X, X_aux.astype(np.float32)], axis=-1)

        if (X_aux_train is not None and X_aux_train.size > 0
                and X_aux_train.shape[-1] > 0):
            if not _use_streaming:
                print(f"Appending {X_aux_train.shape[-1]} auxiliary "
                      f"channel(s) after spectral preprocessing\n")
                X_train = _append_aux(X_train, X_aux_train, "train")
                if X_val.size > 0 and X_aux_val is not None:
                    X_val = _append_aux(X_val, X_aux_val, "val")
                if X_test.size > 0 and X_aux_test is not None:
                    X_test = _append_aux(X_test, X_aux_test, "test")
                print(f"After appending aux channels: "
                      f"X_train={X_train.shape}  "
                      f"X_val={X_val.shape}  X_test={X_test.shape}\n")
                # Free aux arrays now that they are merged into X
                del X_aux_train, X_aux_val, X_aux_test
                gc.collect()
            else:
                # Streaming: keep X_aux_* as separate arrays; the
                # streaming dataset will concat them per patch inside
                # __getitem__. Do NOT free them here.
                print(f"[streaming] Keeping {X_aux_train.shape[-1]} "
                      f"auxiliary channel(s) separate from X; will be "
                      f"concatenated per patch inside DataLoader workers.")


        # ---- Scaling of output (y) with sklearn ---- #
        y_sklearn = None


        if args.y_scaler != "none":
            print(f"Scaling outputs with sklearn {args.y_scaler} scaler...\n")

            # ---- DIAGNOSTIC: raw y stats BEFORE scaling ----
            print("=== Raw y_train stats (before scaling) ===")
            print(f"  shape : {y_train.shape}")
            print(f"  min   : {np.nanmin(y_train)}")
            print(f"  max   : {np.nanmax(y_train)}")
            print(f"  any NaN : {np.isnan(y_train).any()}")
            print(f"  any Inf : {np.isinf(y_train).any()}")
            print(f"  values <= -1 : {int(np.sum(y_train <= -1))}")
            print(f"  values <  0  : {int(np.sum(y_train < 0))}")
            print(f"  values == 0  : {int(np.sum(y_train == 0))}")
            if y_val.size > 0:
                print(f"  y_val min/max: {np.nanmin(y_val)} / {np.nanmax(y_val)}")
            if y_test.size > 0:
                print(f"  y_test min/max: {np.nanmin(y_test)} / {np.nanmax(y_test)}")

            # Zero non-cartilage pixels in y, then clip negatives to 0
            # (required for log-family scalers; see Methods).
            def _mask_to_y_shape(m, y):
                if m.ndim == y.ndim:
                    return m
                if y.ndim == 2 and m.ndim == 3:
                    return m.reshape(m.shape[0], -1)
                if y.ndim == 1 and m.ndim == 3:
                    return m.any(axis=(1, 2))
                if y.ndim == 4 and m.ndim == 3:
                    # spectra patch: broadcast spatial mask over channel
                    return np.broadcast_to(m[..., None], y.shape)
                if y.ndim == 2 and m.ndim == 1:
                    # spectra per-pixel: broadcast (N,) over channel
                    return np.broadcast_to(m[:, None], y.shape)
                return m


            # Replace NaN/Inf first (can appear in bone-removed maps)
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
            if y_val.size > 0:
                y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)
            if y_test.size > 0:
                y_test = np.nan_to_num(y_test, nan=0.0, posinf=0.0, neginf=0.0)

            # Zero out non-mask pixels (they are ignored by the loss anyway)
            m_y_tr = _mask_to_y_shape(m_train, y_train)
            y_train[~m_y_tr] = 0.0
            if y_val.size > 0:
                m_y_va = _mask_to_y_shape(m_val, y_val)
                y_val[~m_y_va] = 0.0
            if y_test.size > 0:
                m_y_te = _mask_to_y_shape(m_test, y_test)
                y_test[~m_y_te] = 0.0

            # For a log transform, also clip negatives (can arise from baseline
            # correction residuals on cartilage pixels)
            if args.y_scaler in ("log", "log_standard"):
                n_neg_tr = int(np.sum(y_train < 0))
                n_neg_va = int(np.sum(y_val < 0)) if y_val.size > 0 else 0
                n_neg_te = int(np.sum(y_test < 0)) if y_test.size > 0 else 0
                if n_neg_tr + n_neg_va + n_neg_te > 0:
                    print(f"  [y_scaler={args.y_scaler}] WARNING: clipping "
                        f"{n_neg_tr} train, {n_neg_va} val, {n_neg_te} test "
                        f"negative y-values to 0. This is a non-invertible "
                        f"transformation. Use --y_scaler power if negative "
                        f"values are physically meaningful.")

                y_train = np.clip(y_train, 0.0, None)
                if y_val.size > 0:
                    y_val = np.clip(y_val, 0.0, None)
                if y_test.size > 0:
                    y_test = np.clip(y_test, 0.0, None)

            print("=== Sanitized y_train stats ===")
            print(f"  min/max : {y_train.min()} / {y_train.max()}")
            print(f"  finite  : {np.isfinite(y_train).all()}")

            # ---- Fit scaler only on masked cartilage pixels ----
            y_train_shape = y_train.shape
            y_val_shape = y_val.shape if y_val.size > 0 else None
            y_test_shape = y_test.shape if y_test.size > 0 else None

            y_sklearn = make_sklearn_scaler(args.y_scaler)
            y_train_masked = y_train[_mask_to_y_shape(m_train, y_train)].reshape(-1, 1)
            y_sklearn.fit(y_train_masked)
            del y_train_masked
            gc.collect()

            # ---- Transform all pixels in memory-safe chunks ----
            def chunked_y_transform(scaler, arr_y, chunk_size=10_000_000):
                if arr_y.size == 0:
                    return arr_y
                arr_flat = arr_y.reshape(-1, 1)
                out_flat = np.empty_like(arr_flat, dtype=np.float32)
                for i in range(0, arr_flat.shape[0], chunk_size):
                    end = min(i + chunk_size, arr_flat.shape[0])
                    # transform returns float64; assigning to out_flat safely downcasts to float32
                    out_flat[i:end] = scaler.transform(arr_flat[i:end])
                return out_flat.reshape(arr_y.shape)

            y_train = chunked_y_transform(y_sklearn, y_train)
            y_train[~_mask_to_y_shape(m_train, y_train)] = 0.0

            if y_val.size > 0:
                y_val = chunked_y_transform(y_sklearn, y_val)
                y_val[~_mask_to_y_shape(m_val, y_val)] = 0.0

            if y_test.size > 0:
                y_test = chunked_y_transform(y_sklearn, y_test)
                y_test[~_mask_to_y_shape(m_test, y_test)] = 0.0

            # ---- DIAGNOSTIC: scaled y stats AFTER scaling ----
            print("=== Scaled y_train stats (after scaling) ===")
            print(f"  min/max : {np.nanmin(y_train)} / {np.nanmax(y_train)}")
            print(f"  any NaN : {np.isnan(y_train).any()}")
            print(f"  any Inf : {np.isinf(y_train).any()}")

            if not np.isfinite(y_train).all():
                raise RuntimeError(
                    "Non-finite values in y_train after scaling — aborting fold."
                )

        # ---- DIAGNOSTIC: per-channel Pearson correlation of X with y (training) ----
        try:
            if X_train.ndim == 4:
                N, P1, P2, C = X_train.shape
                X_flat_diag = X_train.reshape(-1, C)        # (N*P*P, C)

                if y_train.ndim == 1:                       # (N,) scalar per patch
                    y_flat_diag = np.repeat(y_train, P1 * P2)
                elif y_train.ndim == 2 and y_train.shape[1] == P1 * P2:
                    # (N, P*P) flat spatial map per patch
                    y_flat_diag = y_train.reshape(-1)
                elif y_train.ndim == 2:
                    # (N, C_ftir) per-patch spectrum -> scalar per patch
                    y_per_patch = np.nanmean(y_train, axis=1)
                    y_flat_diag = np.repeat(
                        np.nan_to_num(y_per_patch, nan=0.0,
                                    posinf=0.0, neginf=0.0),
                        P1 * P2,
                    )
                elif y_train.ndim == 3:                     # (N, P, P) spatial map
                    y_flat_diag = y_train.reshape(-1)
                elif y_train.ndim == 4:                     # (N, P, P, C_ftir) spectra
                    # Collapse spectral axis to a per-pixel scalar, just for diagnostics.
                    with np.errstate(invalid="ignore"):
                        y_flat_diag = np.nanmean(y_train, axis=-1).reshape(-1)
                else:
                    raise ValueError(
                        f"Correlation diagnostic: unsupported y_train.ndim={y_train.ndim}"
                    )
                if m_train.ndim == 3:
                    m_flat_diag = m_train.reshape(-1).astype(bool)
                else:
                    m_flat_diag = np.repeat(m_train.reshape(-1).astype(bool), P1 * P2)
            else:
                X_flat_diag = X_train
                y_flat_diag = y_train.reshape(-1) if y_train.ndim > 1 else y_train
                m_flat_diag = m_train.reshape(-1).astype(bool) if m_train.ndim > 1 else m_train.astype(bool)
                if m_flat_diag.size != X_flat_diag.shape[0]:
                    m_flat_diag = np.ones(X_flat_diag.shape[0], dtype=bool)

            valid = m_flat_diag & np.isfinite(y_flat_diag) & np.isfinite(X_flat_diag).all(axis=1)
            Xv = X_flat_diag[valid]
            yv = y_flat_diag[valid]

            # Subsample for speed (deterministic w.r.t. args.seed)
            if Xv.shape[0] > 200_000:
                idx = np.random.default_rng(args.seed).choice(
                    Xv.shape[0], 200_000, replace=False
                )
                Xv = Xv[idx]
                yv = yv[idx]


            if Xv.shape[0] >= 2:
                Xv_c = Xv - Xv.mean(axis=0, keepdims=True)
                yv_c = yv - yv.mean()
                num = (Xv_c * yv_c[:, None]).sum(axis=0)
                den = np.linalg.norm(Xv_c, axis=0) * (np.linalg.norm(yv_c) + 1e-12) + 1e-12
                corr = num / den
                print(f"[Corr X vs y (train, masked)] "
                    f"min={corr.min():+.3f}  max={corr.max():+.3f}  "
                    f"mean|r|={np.mean(np.abs(corr)):.3f}  "
                    f"N={Xv.shape[0]:,}")
            else:
                print("[Corr X vs y] too few valid samples to compute")
        except Exception as e:
            print(f"Correlation diagnostic failed: {e}")


        print(f"X_train shape={X_train.shape} \nX_val shape={X_val.shape}   \nX_test shape={X_test.shape}\n")


        # ---- Compute per-bin weights for weighted MSE (on scaled, masked y_train) ----
        y_bins_tensor = None
        y_weights_tensor = None
        if args.loss == "wmse":
            # Build a mask aligned to y_train's shape (handles map and spectra layouts).
            # Supported shapes:
            #   y_train.ndim == 1, m_train.ndim == 3   -> scalar y, spatial mask
            #   y_train.ndim == 2, m_train.ndim == 3   -> flat (N, P*P) y, spatial mask
            #   y_train.ndim == 3, m_train.ndim == 3   -> patch (N, P, P) y, spatial mask
            #   y_train.ndim == 4, m_train.ndim == 3   -> spectra patch (N, P, P, C_out)
            #   y_train.ndim == 2, m_train.ndim == 1   -> spectra per-pixel (N, C_out)
            if m_train.ndim == 3 and y_train.ndim == 1:
                # any-pixel-cartilage per sample
                m_y_tr = m_train.any(axis=(1, 2))                         # (N,)
            elif m_train.ndim == 3 and y_train.ndim == 2:
                # flat spatial map: (N, P*P)
                m_y_tr = m_train.reshape(m_train.shape[0], -1)            # (N, P*P)
            elif m_train.ndim == 3 and y_train.ndim == 3:
                # patch spatial map: (N, P, P)
                m_y_tr = m_train                                          # (N, P, P)
            elif m_train.ndim == 3 and y_train.ndim == 4:
                # spectra patch: broadcast spatial mask over the spectral channel
                m_y_tr = np.broadcast_to(m_train[..., None], y_train.shape)  # (N, P, P, C_out)
            elif m_train.ndim == 1 and y_train.ndim == 2:
                # spectra per-pixel: broadcast (N,) over channel axis
                m_y_tr = np.broadcast_to(m_train[:, None], y_train.shape)    # (N, C_out)
            else:
                # Last-resort: no per-element mask available; weight all values equally.
                m_y_tr = np.ones_like(y_train, dtype=bool)

            y_flat = y_train.reshape(-1)
            m_flat = np.asarray(m_y_tr).reshape(-1).astype(bool)

            if y_flat.size != m_flat.size:
                # Defensive guard: if shapes still disagree, fall back to no weighting
                print(f"[wmse] WARNING: shape mismatch y_flat={y_flat.size} "
                    f"vs m_flat={m_flat.size}; disabling wmse weighting.")
            else:
                y_valid = y_flat[m_flat]
                y_valid = y_valid[np.isfinite(y_valid)]

                if y_valid.size < max(args.wmse_nbins * 2, 20):
                    print(f"[wmse] WARNING: only {y_valid.size} valid y values; "
                        f"disabling wmse weighting (fallback to unweighted MSE).")
                else:
                    nbins = args.wmse_nbins
                    hist, edges = np.histogram(y_valid, bins=nbins)
                    freq = hist.astype(np.float64) / max(hist.sum(), 1)
                    w = 1.0 / (freq + 1e-6) ** args.wmse_power
                    w_mean_pix = (w * freq).sum() / max(freq.sum(), 1e-12)
                    w = w / max(w_mean_pix, 1e-12)
                    w = np.clip(w, 0.1, 20.0)

                    y_bins_tensor = edges[1:-1].astype(np.float32)
                    y_weights_tensor = w.astype(np.float32)
                    print(f"[wmse] bins={nbins}, weight min/max = "
                        f"{w.min():.3f}/{w.max():.3f}")



        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Initialize metrics container ----
        metrics = {
            "run_tag": run_tag,
            "model": args.model,
            "test_mae_scaled": None,
            "test_rmse_scaled": None,
            "test_r2_scaled":  None,
            "train_mae_scaled": None,
            "train_rmse_scaled": None,
            "train_r2_scaled":  None,
            "test_mae_unscaled_per_sample": {},
            "test_rmse_unscaled_per_sample": {},
            "test_r2_unscaled_per_sample":  {},
            "test_local_std_ratio_per_sample": {},
            # Spectra-mode only: R^2 decomposition into global / per-pixel
            # spectral shape / per-channel spatial. Empty in map mode.
            "test_r2_global_per_sample":   {},
            "test_r2_spectral_per_sample": {},
            "test_r2_spatial_per_sample":  {},
        }


        if not _is_bo:
            # ---- Save args to text and JSON files ----
            args_txt_path = os.path.join(results_dir, "args.txt")
            with open(args_txt_path, "w") as f:
                f.write(f"Short tag: {short_tag}\n")
                f.write(f"Full run_tag: {run_tag}\n")
                f.write("=" * 60 + "\n")
                # ---- Methods provenance (leak-free protocol) ----
                f.write("PROTOCOL\n")
                f.write("--------\n")
                f.write(
                    "All unsupervised preprocessing steps (baseline "
                    "correction, x-scaler, PCA/PLSR-DR, y-scaler) are "
                    "fitted on training cartilage pixels only. "
                    "Validation and test samples are transformed with "
                    "these fitted models.\n"
                    "Per-spectrum normalisation (SNV/L2) and SG "
                    "derivatives are computed independently per spectrum "
                    "and therefore carry no information across samples.\n"
                    "Bayesian hyperparameter optimisation uses "
                    "validation-set quantities only; the test sample is "
                    "not seen by BO. The validation R^2 used by BO is "
                    "the masked R^2 computed in physical units (after "
                    "inverse y-scaling) and without shift-tolerance "
                    "alignment, so it is scaler-independent. In "
                    "target_mode=spectra, this R^2 is computed on the "
                    "flattened (pixel x wavenumber) array and therefore "
                    "conflates per-pixel spectral shape with inter-pixel "
                    "mean differences; the decomposed test metrics "
                    "R^2_global, R^2_spectral, and R^2_spatial are "
                    "reported alongside as more interpretable "
                    "quantities. The optional blur penalty is computed "
                    "in physical units on a 2D mean-over-channels "
                    "summary.\n"
                    "Per-image PCA-of-the-cube auxiliary channels are "
                    "disabled in this build because they would fit a "
                    "basis on the held-out sample; the mean-of-cube "
                    "auxiliary channel is retained because it is fixed "
                    "per pixel.\n"
                    "Fold pixels (non-cartilage) are filled with a "
                    "constant value. --nonmask_fill=patch_mean is "
                    "excluded from the BO search space because it "
                    "injects a sample-conditioned constant.\n"
                )


                f.write("=" * 60 + "\n")
                for k, v in sorted(vars(args).items()):
                    f.write(f"{k}: {v}\n")
            print(f"Arguments saved to: {args_txt_path}")

            # Also save as JSON for easy machine reading later
            args_json_path = os.path.join(results_dir, "args.json")
            with open(args_json_path, "w") as f:
                json.dump(
                    {"short_tag": short_tag, "run_tag": run_tag, **vars(args)},
                    f, indent=2, default=str,
                )
            print(f"Arguments saved to: {args_json_path}")

            cmd_path = os.path.join(results_dir, "command.txt")
            with open(cmd_path, "w") as f:
                f.write(" ".join(sys.argv) + "\n")


        # Keep test patch arrays for per-sample tile visualization
        y_test_raw = y_test.copy() if y_test.size else None
        m_test_raw = m_test.copy() if m_test.size else None


        # ---- Build and train model ---- #
        if args.model == "plsr":
            if args.avg_mode == "patch":
                N_tr = X_train.shape[0]
                X_train_plsr = X_train.reshape(N_tr, -1)
                N_te = X_test.shape[0]
                X_test_plsr = X_test.reshape(N_te, -1)
            else:
                X_train_plsr = X_train
                X_test_plsr = X_test

            if args.target_mode == "spectra":
                plsr = PLSRWrapperSpectra(n_components=args.plsr_components)
                plsr.fit(X_train_plsr, y_train)
            else:
                plsr = PLSRWrapper(n_components=args.plsr_components)
                plsr.fit(X_train_plsr, y_train.ravel())

            print("PLSR fitted.")
            
            # Evaluate PLSR on the validation set so that the Optuna objective
            # receives a defined score even when target_mode='map' + model='plsr'.
            if X_val.size > 0:
                X_val_plsr = X_val.reshape(X_val.shape[0], -1) if args.avg_mode == "patch" else X_val
                y_val_pred_scaled = plsr.predict(X_val_plsr)
                
                # Align mask shape for validation (same logic as test mask)
                if y_val.ndim == 1 and m_val.ndim == 3:
                    m_val_aligned = m_val.any(axis=(1, 2))
                elif y_val.ndim == 2 and m_val.ndim == 3:
                    m_val_aligned = m_val.reshape(m_val.shape[0], -1)
                elif y_val.ndim == 4 and m_val.ndim == 3:
                    m_val_aligned = np.broadcast_to(m_val[..., None], y_val.shape)
                elif y_val.ndim == 2 and m_val.ndim == 1:
                    m_val_aligned = np.broadcast_to(m_val[:, None], y_val.shape)
                else:
                    m_val_aligned = m_val
                    
                y_val_masked = y_val.ravel()[m_val_aligned.ravel().astype(bool)]
                p_val_masked = y_val_pred_scaled.ravel()[m_val_aligned.ravel().astype(bool)]
                
                if y_val_masked.size > 0:
                    val_mae, _, _, val_r2 = compute_metrics(y_val_masked, p_val_masked)
                    metrics["val_r2_best"] = float(val_r2) if not np.isnan(val_r2) else -1e9
                    # Use RMSE or MSE to be consistent with regression-loss semantics elsewhere
                    _, val_mse, _, _ = compute_metrics(y_val_masked, p_val_masked)
                    metrics["val_loss_best"] = float(val_mse) if not np.isnan(val_mse) else 1e9
                    metrics["val_r2_minus_blur_best"] = metrics["val_r2_best"]
                    # PLSR does not have a per-epoch blur history. We compute a single
                    # blur value from the validation prediction and use that for the
                    # Pareto objective. If we cannot compute it (no spatial structure
                    # in the output), we mark it absent so the Pareto objective will
                    # fall back to 1e9 — same as before. PLSR is not in the BO search
                    # space in this study, so this branch is a defensive guard, not a
                    # performance-critical path.
                    try:
                        if (y_val.ndim >= 2 and m_val.ndim == 3
                                and y_val_pred_scaled is not None):
                            # Reshape the flat scaled prediction back to a spatial map per patch
                            if args.target_mode == "map" and args.out_avg_mode == "patch":
                                pred_patches = y_val_pred_scaled.reshape(
                                    -1, args.out_patch, args.out_patch)
                                gt_patches = y_val.reshape(
                                    -1, args.out_patch, args.out_patch)
                                ratios = []
                                for i in range(pred_patches.shape[0]):
                                    if m_val[i].any():
                                        r = local_std_ratio(
                                            gt_patches[i], pred_patches[i],
                                            m_val[i].astype(bool), k=5)
                                        if np.isfinite(r):
                                            ratios.append(abs(1.0 - r))
                                if ratios:
                                    metrics["val_blur_pen_best"] = float(np.mean(ratios))
                    except Exception as _e:
                        print(f"[PLSR] val_blur_pen_best computation failed: {_e}")
            else:
                # No validation set: fall back to training metrics so BO
                # does not see -1e9 for every PLSR trial.
                y_train_pred = plsr.predict(X_train_plsr)
                if y_train.ndim == 1 and m_train.ndim == 3:
                    mtr = m_train.any(axis=(1, 2))
                else:
                    mtr = np.ones_like(y_train, dtype=bool)
                ytr_m = y_train.ravel()[mtr.ravel().astype(bool)]
                ptr_m = y_train_pred.ravel()[mtr.ravel().astype(bool)]
                if ytr_m.size > 0:
                    tr_mae, _, _, tr_r2 = compute_metrics(ytr_m, ptr_m)
                    metrics["val_r2_best"] = float(tr_r2) if np.isfinite(tr_r2) else -1e9
                    metrics["val_loss_best"] = float(tr_mae) if np.isfinite(tr_mae) else 1e9
                    metrics["val_r2_minus_blur_best"] = metrics["val_r2_best"]
                    print("[PLSR] no val set; using train R2 as BO fallback.")

            # -------------------------------------------------------------------

            # Predict on test set (output is scaled because input y was scaled)
            y_pred_scaled = plsr.predict(X_test_plsr)

            if args.target_mode == "map":
                if args.out_avg_mode == "patch":
                    y_pred_scaled = y_pred_scaled.reshape(-1, args.out_patch, args.out_patch)
                elif args.out_avg_mode == "flat":
                    y_pred_scaled = y_pred_scaled.reshape(-1, args.out_patch * args.out_patch)

            # Align mask shape to y shape (handles map and spectra layouts)
            if y_test.ndim == 1 and m_test.ndim == 3:
                # map: scalar y, spatial mask -> any-pixel-valid per sample
                m_test_aligned = m_test.any(axis=(1, 2))
            elif y_test.ndim == 2 and m_test.ndim == 3:
                # map: flat y, spatial mask -> (N, P*P)
                m_test_aligned = m_test.reshape(m_test.shape[0], -1)
            elif y_test.ndim == 4 and m_test.ndim == 3:
                # spectra patch: y (N, P, P, C_out), mask (N, P, P)
                # broadcast spatial mask over spectral channel
                m_test_aligned = np.broadcast_to(
                    m_test[..., None], y_test.shape
                )
            elif y_test.ndim == 2 and m_test.ndim == 1:
                # spectra per-pixel: y (N, C_out), mask (N,)
                m_test_aligned = np.broadcast_to(
                    m_test[:, None], y_test.shape
                )
            else:
                m_test_aligned = m_test


            y_test_flat = y_test.ravel()
            y_pred_flat = y_pred_scaled.ravel()
            m_flat = m_test_aligned.ravel().astype(bool)

            y_masked = y_test_flat[m_flat]
            p_masked = y_pred_flat[m_flat]


            mae, mse, rmse, r2 = compute_metrics(y_masked, p_masked)
            print(f"[Test - masked] MAE = {mae:.2e} RMSE = {rmse:.2e} R2 = {r2:.3f}")

            metrics["test_mae_scaled"]  = float(mae)
            metrics["test_rmse_scaled"] = float(rmse)
            metrics["test_r2_scaled"]   = float(r2)

            if not _is_bo:
                plsr_path = os.path.join(results_dir, "model_plsr.pkl")
                with open(plsr_path, "wb") as f:
                    pickle.dump(plsr, f)
                print(f"PLSR model saved to: {plsr_path}")


        else:
            # ---- Determine output dimensionality (spectra-aware) ----
            if y_train.ndim == 1:
                out_dim = 1
                C_out_model = 1
            else:
                out_dim = int(np.prod(y_train.shape[1:]))
                C_out_model = y_train.shape[-1]

            # ---- Compute the actual channel count the model will see ----
            # Under streaming, X_train still holds raw spectra (no x_scaler
            # transform applied yet) and the aux channels have NOT been
            # concatenated. The streaming dataset will apply x_scaler and
            # concatenate aux at runtime, so we must reconstruct the final
            # channel count here. This is also what the GAN's discriminator
            # conditioning will see, so it must be correct.
            if _use_streaming:
                # Build a temporary streaming dataset matching the real one and probe
                # one sample to get the true channel count the model will receive.
                _probe_ds = UHsiStreamingDataset(
                    X_raw=X_train[:1], y_raw=y_train[:1],
                    m_out=m_train[:1],
                    X_aux=(X_aux_train[:1] if X_aux_train is not None else None),
                    x_scaler=x_sklearn, y_scaler=None,
                    smooth_x=_stream_smooth_x,
                    sg_window=args.sg_window, sg_poly=args.sg_poly,
                    spectrum_norm=_stream_spectrum_norm,
                    der_mode=_stream_der_mode,
                    delta_wl=float(wls[1] - wls[0]) if len(wls) > 1 else 1.0,
                    dr_model=dr_model, model_type=args.model,
                )
                x_probe, _, _ = _probe_ds[0]
                in_ch_model = int(x_probe.shape[0])  # C in (C, P, P)
                print(f"[streaming] Model input channels probed from dataset "
                    f"= {in_ch_model}")
                del _probe_ds, x_probe

            else:
                in_ch_model = int(X_train.shape[-1])


            if args.target_mode == "map":

                # ----- MAP MODE: scalar-output models (existing behavior) -----
                if args.model == "mlp":
                    in_dim = X_train.shape[1]
                    model = MLP(in_dim, hidden=args.mlp_hidden, dropout=args.dropout)
                elif args.model == "cnn1d":
                    in_ch = X_train.shape[1]
                    model = CNN1D(in_ch, hidden=args.cnn1d_hidden,
                                  n_layers=args.cnn1d_layers, dropout=args.dropout)
                elif args.model == "cnn":
                    in_ch = in_ch_model
                    model = CNN2D(in_ch, patch=args.patch,
                                  hidden=args.cnn_hidden, dropout=args.dropout)
                elif args.model == "unet_simple":
                    in_ch = in_ch_model
                    model = UNet2D_simple(
                        in_ch=in_ch, base_ch=args.unet_base_ch,
                        dropout=args.dropout,
                        out_mode=args.out_avg_mode,
                        out_patch=args.out_patch,
                        depth=args.unet_depth,
                    )
                elif args.model == "unet":
                    in_ch = in_ch_model
                    model = UNet2D(
                        in_ch=in_ch, base_ch=args.unet_base_ch,
                        dropout=args.dropout,
                        out_mode=args.out_avg_mode,
                        out_patch=args.out_patch,
                        depth=args.unet_depth, use_se=False,
                    )
                elif args.model == "pretrained_unet":
                    in_ch = in_ch_model
                    weights = None if args.pretrained_weights == "none" else args.pretrained_weights
                    model = PretrainedUNet(
                        in_ch=in_ch,
                        out_ch=1,
                        encoder_name=args.pretrained_encoder,
                        encoder_weights=weights,
                        out_mode=args.out_avg_mode,
                        out_patch=args.out_patch,
                        dropout=args.dropout,
                        target_mode="map",
                    )
                    print(f"[pretrained] {args.pretrained_encoder} "
                        f"(weights={weights}, in_ch={in_ch})")
                elif args.model == "multi_unet":
                    if not args.map_fields:
                        raise ValueError(
                            "--model multi_unet requires --map_fields with "
                            "at least one map name."
                        )
                    in_ch = in_ch_model
                    n_maps = len(args.map_fields)
                    model = MultiMapUNet2D(
                        in_ch=in_ch, n_maps=n_maps,
                        base_ch=args.unet_base_ch,
                        dropout=args.dropout,
                        out_mode=args.out_avg_mode,
                        out_patch=args.out_patch,
                        depth=args.unet_depth,
                        use_se=False,
                        share_head=args.multi_unet_share_head,
                    )
                    print(f"[multi_unet] predicting {n_maps} maps: "
                          f"{args.map_fields}  (in_ch={in_ch})")
                else:
                    raise ValueError(f"Unknown model in map mode: {args.model}")


            else:
                # ----- SPECTRA MODE: spectra-aware models -----
                if args.model == "unet":
                    in_ch = in_ch_model
                    # Reuse the SAME residual+SE UNet2D used for map mode.
                    # Only the head and output layout change (out_ch=C_ftir,
                    # BHWC). All other hyperparameters are shared with map
                    # mode, matching the manuscript's claim that the
                    # architecture is target-agnostic.
                    model = UNet2D(
                        in_ch=in_ch,
                        base_ch=args.unet_base_ch,
                        dropout=args.dropout,
                        out_mode=args.out_avg_mode,
                        out_patch=args.out_patch,
                        depth=args.unet_depth,
                        use_se=False,            # keep parity with map mode
                        out_ch=C_out_model,      # = C_ftir from y_train
                        target_mode="spectra",
                    )
                elif args.model == "mlp":
                    assert X_train.ndim == 2, \
                        "spectra mode + 'mlp' requires --patch 1 --avg_mode avg"
                    model = SpectrumMLP(
                        c_in=X_train.shape[1], c_out=C_out_model,
                        hidden=args.mlp_hidden, dropout=args.dropout,
                    )
                elif args.model == "cnn1d":
                    assert X_train.ndim == 2, \
                        "spectra mode + 'cnn1d' requires --patch 1 --avg_mode avg"
                    model = SpectralCNN1D(
                        c_in=X_train.shape[1], c_out=C_out_model,
                        hidden=args.cnn1d_hidden,
                        n_layers=args.cnn1d_layers,
                        dropout=args.dropout,
                    )
                elif args.model == "pretrained_unet":
                    in_ch = in_ch_model
                    weights = None if args.pretrained_weights == "none" else args.pretrained_weights
                    model = PretrainedUNet(
                        in_ch=in_ch,
                        out_ch=C_out_model,
                        encoder_name=args.pretrained_encoder,
                        encoder_weights=weights,
                        out_mode=args.out_avg_mode,
                        out_patch=args.out_patch,
                        dropout=args.dropout,
                        target_mode="spectra",
                    )
                    print(f"[pretrained spectra] {args.pretrained_encoder} "
                        f"(weights={weights}, in_ch={in_ch}, out_ch={C_out_model})")
                else:
                    raise ValueError(
                        f"Model '{args.model}' is not yet supported in target_mode=spectra"
                    )

            # ---- Multi-GPU with DataParallel ----
            n_gpus = torch.cuda.device_count()
            print(f"Available GPUs: {n_gpus}")
            # Surface any pending async CUDA error from a previous trial
            # NOW (synchronously), so it is attributed to that trial and
            # does not crash this trial at model.to(device).
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except RuntimeError as _e:
                    print(f"[run_one_fold] Pre-existing CUDA error detected "
                          f"before model.to(device): {_e}")
                    raise
            try:
                model = model.to(device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except RuntimeError as _e:
                msg = str(_e).lower()
                if any(k in msg for k in ("cuda", "cublas", "cudnn",
                                          "illegal memory", "out of memory",
                                          "device-side assert")):
                    print(f"[run_one_fold] CUDA failure on model.to(device): {_e}")
                raise


            if _use_streaming:
                # Build raw-patch streaming datasets. SG/SNV/der were
                # already applied in the parallel loader, so we tell the
                # streaming dataset NOT to redo them (see _stream_*).
                # x_sklearn and dr_model (if any) are passed in and
                # applied per patch inside __getitem__.
                delta_wl = (float(wls[1] - wls[0])
                            if len(wls) > 1 else 1.0)

                # If aux channels are empty / disabled, set them to None
                # so the streaming dataset's __getitem__ has nothing to
                # concatenate.
                if not (X_train.ndim == 4 and (
                        X_aux_train is not None
                        and X_aux_train.size > 0
                        and X_aux_train.shape[-1] > 0)):
                    X_aux_train = None
                    X_aux_val   = None
                    X_aux_test  = None

                train_ds = UHsiStreamingDataset(
                    X_raw=X_train, y_raw=y_train,
                    m_out=m_train, X_aux=X_aux_train,
                    x_scaler=x_sklearn, y_scaler=None,
                    smooth_x=_stream_smooth_x,
                    sg_window=args.sg_window, sg_poly=args.sg_poly,
                    spectrum_norm=_stream_spectrum_norm,
                    der_mode=_stream_der_mode, delta_wl=delta_wl,
                    dr_model=dr_model, model_type=args.model,
                )
                val_ds = (UHsiStreamingDataset(
                    X_raw=X_val, y_raw=y_val,
                    m_out=m_val, X_aux=X_aux_val,
                    x_scaler=x_sklearn, y_scaler=None,
                    smooth_x=_stream_smooth_x,
                    sg_window=args.sg_window, sg_poly=args.sg_poly,
                    spectrum_norm=_stream_spectrum_norm,
                    der_mode=_stream_der_mode, delta_wl=delta_wl,
                    dr_model=dr_model, model_type=args.model,
                ) if X_val.size else None)
                test_ds = (UHsiStreamingDataset(
                    X_raw=X_test, y_raw=y_test,
                    m_out=m_test, X_aux=X_aux_test,
                    x_scaler=x_sklearn, y_scaler=None,
                    smooth_x=_stream_smooth_x,
                    sg_window=args.sg_window, sg_poly=args.sg_poly,
                    spectrum_norm=_stream_spectrum_norm,
                    der_mode=_stream_der_mode, delta_wl=delta_wl,
                    dr_model=dr_model, model_type=args.model,
                ) if X_test.size else None)
                # IMPORTANT: do NOT delete X_train/y_train/m_train etc.;
                # the streaming datasets hold references to them.
            else:
                train_ds = UHsiDataset(X_train, y_train, m_train,
                                       model_type=args.model)
                val_ds   = (UHsiDataset(X_val,   y_val,   m_val,
                                        model_type=args.model)
                            if X_val.size else None)
                test_ds  = (UHsiDataset(X_test,  y_test,  m_test,
                                        model_type=args.model)
                            if X_test.size else None)

                del X_train, y_train, m_train
                del X_val, y_val, m_val
                del X_test, y_test, m_test
                gc.collect()


            # Pinned host memory is fragile across many short BO trials
            # because the pluggable host allocator has been observed to fault
            # on context teardown. Disable pin_memory during BO and re-enable
            # it for the final, full training run.
            pin = torch.cuda.is_available() and not getattr(args, "bo", False)
            # Cap workers: use SLURM allocation if available, else sensible default
            _slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
            if _slurm_cpus.isdigit():
                _max_workers = max(int(_slurm_cpus) - 2, 1)
            else:
                _max_workers = 0

            print(f"DataLoader num_workers = {_max_workers}")

            train_loader = DataLoader(train_ds, batch_size=args.batch,
                                    shuffle=True, drop_last=False,
                                    num_workers=_max_workers,
                                    pin_memory=pin,
                                    persistent_workers=_max_workers > 0,
                                    prefetch_factor=(2 if _max_workers > 0
                                                     else None),
                                    worker_init_fn=getattr(args, "_worker_seed_init", None))
            val_loader = (DataLoader(val_ds, batch_size=args.batch,
                                    shuffle=False, drop_last=False,
                                    num_workers=_max_workers,
                                    pin_memory=pin,
                                    persistent_workers=_max_workers > 0,
                                    prefetch_factor=(2 if _max_workers > 0
                                                     else None),
                                    worker_init_fn=getattr(args, "_worker_seed_init", None))
                          if val_ds is not None else None)
            test_loader = (DataLoader(test_ds, batch_size=args.batch,
                                    shuffle=False, drop_last=False,
                                    num_workers=_max_workers,
                                    pin_memory=pin,
                                    persistent_workers=_max_workers > 0,
                                    prefetch_factor=(2 if _max_workers > 0
                                                     else None),
                                    worker_init_fn=getattr(args, "_worker_seed_init", None))
                           if test_ds is not None else None)



            scheduler_kwargs = {
                "T_max": args.cosine_Tmax,
                "eta_min": args.cosine_min_lr,
                "factor": args.plateau_factor,
                "patience": args.plateau_patience,
                "min_lr": args.plateau_min_lr,
                "threshold": args.plateau_threshold,
            }

            



            def _set_encoder_trainable_keep_first_conv(m, trainable, keep_first_conv_trainable=True):
                enc = getattr(getattr(m, "net", m), "encoder", None)
                if enc is None:
                    return

                # Freeze or unfreeze whole encoder
                for p in enc.parameters():
                    p.requires_grad = trainable

                # Re-enable the first Conv2d if desired
                if keep_first_conv_trainable:
                    first_conv = None
                    for module in enc.modules():
                        if isinstance(module, nn.Conv2d):
                            first_conv = module
                            break

                    if first_conv is not None:
                        for p in first_conv.parameters():
                            p.requires_grad = True

            if args.model == "pretrained_unet" and args.freeze_encoder_epochs > 0:
                # ---- Phase 1: train decoder + head only ----
                print(f"[fine-tune] Phase 1: freezing encoder for "
                    f"{args.freeze_encoder_epochs} epochs")
                _set_encoder_trainable_keep_first_conv(
                    model,
                    trainable=False,
                    keep_first_conv_trainable=True,
                )

                history1 = train_torch_model(
                    model, train_loader, val_loader, device,
                    epochs=args.freeze_encoder_epochs, lr=args.lr,
                    scheduler_type=args.scheduler,
                    scheduler_kwargs=scheduler_kwargs,
                    grad_clip=0.5,
                    patience_early_stop=0,           # disable ES during warmup
                    smooth_weight=args.smooth_weight, max_shift=args.max_shift,
                    loss_type=args.loss, huber_delta=args.huber_delta,
                    y_bins=y_bins_tensor, y_weights=y_weights_tensor,
                    grad_weight=args.grad_weight, lap_weight=args.lap_weight,
                    var_weight=args.var_weight,
                    var_kernel=getattr(args, "var_kernel", 5),
                    fft_weight=args.fft_weight,
                    spatial_loss_summary=getattr(args, "spatial_loss_summary", "mean"),
                    spatial_loss_bands=getattr(args, "spatial_loss_bands", None),
                    y_scaler=y_sklearn,
                    y_scaler_kind=args.y_scaler,
                    per_channel_spatial=getattr(args, "per_channel_spatial", False),
                    spectral_smooth_weight=getattr(args, "spectral_smooth_weight", 0.0),
                )



                # ---- Phase 2: unfreeze encoder, fine-tune at lower LR ----
                print(f"[fine-tune] Phase 2: unfreezing encoder, "
                    f"lr={args.lr * args.finetune_lr_mult:.2e}")
                _set_encoder_trainable_keep_first_conv(
                    model,
                    trainable=True,
                    keep_first_conv_trainable=True,
                )

                epochs_phase2 = max(1, args.epochs - args.freeze_encoder_epochs)
                history2 = train_torch_model(
                    model, train_loader, val_loader, device,
                    epochs=epochs_phase2, lr=args.lr * args.finetune_lr_mult,
                    scheduler_type=args.scheduler,
                    scheduler_kwargs=scheduler_kwargs,
                    grad_clip=0.5,
                    patience_early_stop=args.patience_early_stop,
                    smooth_weight=args.smooth_weight, max_shift=args.max_shift,
                    loss_type=args.loss, huber_delta=args.huber_delta,
                    y_bins=y_bins_tensor, y_weights=y_weights_tensor,
                    grad_weight=args.grad_weight, lap_weight=args.lap_weight,
                    var_weight=args.var_weight,
                    var_kernel=getattr(args, "var_kernel", 5),
                    fft_weight=args.fft_weight,
                    spatial_loss_summary=getattr(args, "spatial_loss_summary", "mean"),
                    spatial_loss_bands=getattr(args, "spatial_loss_bands", None),
                    y_scaler=y_sklearn,
                    y_scaler_kind=args.y_scaler,
                    per_channel_spatial=getattr(args, "per_channel_spatial", False),
                    spectral_smooth_weight=getattr(args, "spectral_smooth_weight", 0.0),
                )


                # Free training-time DataLoaders before evaluation, but KEEP train_ds and
                # test_ds: train_ds is reused for the train-evaluation pass, and test_ds
                # is the underlying dataset of test_loader. val_ds is no longer needed.
                try:
                    del train_loader
                except NameError:
                    pass
                try:
                    del val_loader
                except NameError:
                    pass
                if 'val_ds' in locals():
                    del val_ds
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


                # Concatenate histories so the loss curve plot shows both phases
                history = {
                    "epoch": list(range(1, len(history1["train_loss"]) + len(history2["train_loss"]) + 1)),
                    "train_loss": list(history1["train_loss"]) + list(history2["train_loss"]),
                    "val_loss":   list(history1["val_loss"])   + list(history2["val_loss"]),
                    "val_select": list(history1["val_select"]) + list(history2["val_select"]),
                    "train_r2":   list(history1.get("train_r2", [])) + list(history2.get("train_r2", [])),
                    "val_r2":     list(history1.get("val_r2",   [])) + list(history2.get("val_r2",   [])),
                    "lr":         list(history1["lr"])         + list(history2["lr"]),
                }



            else:
                if getattr(args, "gan_mode", "none") != "none":
                    print(f"[GAN] mode={args.gan_mode}  "
                          f"lambda_l1={args.gan_lambda_l1}  "
                          f"lambda_adv={args.gan_lambda_adv}  "
                          f"lambda_inv={args.gan_lambda_cycle}  "  # inverse-consistency, NOT CycleGAN cycle
                          f"shift_aware={args.gan_shift_aware} "
                          f"(max_shift={args.max_shift})")
                    with _section("train_torch_model_gan"):
                        history = train_torch_model_gan(
                            model, train_loader, val_loader, device,
                            epochs=args.epochs, lr=args.lr,
                            scheduler_type=args.scheduler,
                            scheduler_kwargs=scheduler_kwargs,
                            grad_clip=0.5,
                            patience_early_stop=args.patience_early_stop,
                            max_shift=args.max_shift,
                            huber_delta=args.huber_delta,
                            y_bins=y_bins_tensor,
                            y_weights=y_weights_tensor,
                            grad_weight=args.grad_weight,
                            lap_weight=args.lap_weight,
                            var_weight=args.var_weight,
                            var_kernel=getattr(args, "var_kernel", 5),
                            fft_weight=args.fft_weight,
                            spatial_loss_summary=getattr(args, "spatial_loss_summary", "mean"),
                            spatial_loss_bands=getattr(args, "spatial_loss_bands", None),
                            y_scaler=y_sklearn,
                            y_scaler_kind=args.y_scaler,
                            per_channel_spatial=getattr(args, "per_channel_spatial", False),
                            spectral_smooth_weight=getattr(args, "spectral_smooth_weight", 0.0),
                            gan_mode=args.gan_mode,
                            gan_lambda_l1=args.gan_lambda_l1,
                            gan_lambda_adv=args.gan_lambda_adv,
                            gan_lambda_cycle=args.gan_lambda_cycle,
                            gan_lambda_identity=args.gan_lambda_identity,
                            gan_d_base_ch=args.gan_d_base_ch,
                            gan_d_n_layers=args.gan_d_n_layers,
                            gan_d_lr=args.gan_d_lr,
                            gan_d_beta1=args.gan_d_beta1,
                            gan_loss_type=args.gan_loss_type,
                            gan_d_update_every=args.gan_d_update_every,
                            gan_shift_aware=args.gan_shift_aware,
                            in_ch=None,
                            target_mode=args.target_mode,
                            loss_type=args.loss,
                        )
                else:
                    with _section("train_torch_model"):
                        history = train_torch_model(
                            model, train_loader, val_loader, device,
                            epochs=args.epochs, lr=args.lr,
                            scheduler_type=args.scheduler,
                            scheduler_kwargs=scheduler_kwargs,
                            grad_clip=0.5,
                            patience_early_stop=args.patience_early_stop,
                            smooth_weight=args.smooth_weight,
                            max_shift=args.max_shift,
                            loss_type=args.loss,
                            huber_delta=args.huber_delta,
                            y_bins=y_bins_tensor,
                            y_weights=y_weights_tensor,
                            grad_weight=args.grad_weight,
                            lap_weight=args.lap_weight,
                            var_weight=args.var_weight,
                            var_kernel=getattr(args, "var_kernel", 5),
                            fft_weight=args.fft_weight,
                            spatial_loss_summary=getattr(args, "spatial_loss_summary", "mean"),
                            spatial_loss_bands=getattr(args, "spatial_loss_bands", None),
                            y_scaler=y_sklearn,
                            y_scaler_kind=args.y_scaler,
                            per_channel_spatial=getattr(args, "per_channel_spatial", False),
                            spectral_smooth_weight=getattr(args, "spectral_smooth_weight", 0.0),
                        )



                # Free training-time DataLoaders before evaluation, but KEEP train_ds and
                # test_ds: train_ds is reused for the train-evaluation pass, and test_ds
                # is the underlying dataset of test_loader. val_ds is no longer needed.
                try:
                    del train_loader
                except NameError:
                    pass
                try:
                    del val_loader
                except NameError:
                    pass
                if 'val_ds' in locals():
                    del val_ds
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()



            # ---- Save loss history to disk ----
            hist_path = os.path.join(results_dir, "loss_history.json")

            # ---- Inject the canonical, leak-free BO signals ----
            # history["val_r2"] is the masked R^2 in physical units (after
            # inverse y-scaling), comparable across non-linear y-scalers.
            # history["val_blur_pen"] is also computed in physical units so
            # the combined objective is internally consistent.
            if "val_r2" in history and history["val_r2"]:
                valid_val_r2 = [v for v in history["val_r2"]
                                if v is not None and np.isfinite(v)]
                metrics["val_r2_best"] = (float(np.max(valid_val_r2))
                                        if valid_val_r2 else -1e9)

                if ("val_blur_pen" in history
                        and history["val_blur_pen"]):
                    # Scalarized variant (kept for back-compat with bo_metric=
                    # 'val_r2_minus_blur'). Same definition as before.
                    combined_scores = []
                    # Per-epoch blur values, used for the Pareto objective below.
                    per_epoch_blur = []
                    for r2, blur in zip(history["val_r2"],
                                        history["val_blur_pen"]):
                        if r2 is not None and np.isfinite(r2):
                            score = r2 - getattr(
                                args, "bo_blur_penalty", 0.3) * blur
                            combined_scores.append(score)
                        if blur is not None and np.isfinite(blur):
                            per_epoch_blur.append(float(blur))
                    metrics["val_r2_minus_blur_best"] = (
                        float(np.max(combined_scores))
                        if combined_scores else -1e9
                    )
                    # Best (smallest) blur on the same epochs the BO can see.
                    # This is the second Pareto objective. We use min because
                    # 'minimize' is the NSGA-II direction.
                    metrics["val_blur_pen_best"] = (
                        float(np.min(per_epoch_blur))
                        if per_epoch_blur else 1e9
                    )


            if "val_select" in history and history["val_select"]:
                valid_val_loss = [v for v in history["val_select"]
                                  if v is not None and np.isfinite(v)]
                metrics["val_loss_best"] = (float(np.min(valid_val_loss))
                                            if valid_val_loss else 1e9)
            # ------------------------------------------------------------------------

            # Spectra-mode per-pixel R^2 (scaler-independent, physical
            # units). Best epoch is the MAX over the training history.
            # Left absent (defaults to -1e9 in _score_from_metrics) for
            # map mode, where the metric is undefined and BO is not
            # allowed to select it anyway.
            if "val_r2_spectral" in history and history["val_r2_spectral"]:
                valid_val_r2s = [
                    v for v in history["val_r2_spectral"]
                    if v is not None and np.isfinite(v)
                ]
                metrics["val_r2_spectral_best"] = (
                    float(np.max(valid_val_r2s))
                    if valid_val_r2s else -1e9
                )

            safe_hist = {
                k: [None if (isinstance(v, float) and not np.isfinite(v)) else v
                    for v in vals]
                for k, vals in history.items()
            }
            with open(hist_path, "w") as f:
                json.dump(safe_hist, f, indent=2)
            print(f"Loss history saved to: {hist_path}")

            # ---- Plot loss curves (skip for BO trials to save disk + time) ----
            if not _is_bo:
                plot_loss_curves(history, results_dir,
                                 run_tag=run_tag, fold_tag=fold_tag)


            # ---- Save model checkpoint (skip for BO trials) ----
            if not _is_bo:
                checkpoint_path = os.path.join(results_dir, "model_checkpoint.pt")

                # Unwrap DataParallel for clean checkpoint
                model_to_save = model.module if hasattr(model, 'module') else model
                args_to_save = {k: v for k, v in vars(args).items()
                                if not k.startswith("_")}
                checkpoint = {
                    "model_state_dict": model_to_save.state_dict(),
                    "args": args_to_save,
                    "run_tag": run_tag,
                }


                # Save scaler states if they exist
                if x_sklearn is not None:
                    checkpoint["x_scaler"] = pickle.dumps(x_sklearn)
                if y_sklearn is not None:
                    checkpoint["y_scaler"] = pickle.dumps(y_sklearn)
                if dr_model is not None:
                    checkpoint["dr_model"] = pickle.dumps(dr_model)

                torch.save(checkpoint, checkpoint_path)
                print(f"Model checkpoint saved to: {checkpoint_path}")
            else:
                print("[BO] Skipping model checkpoint save.")

            if not _is_bo:
                # ===== TRAIN EVALUATION (sanity check: can model fit training data?) =====
                # Subsample to at most 5000 patches for the diagnostic scatter.
                # The full training pass adds little information beyond this and
                # can take minutes per fold on large datasets.
                n_train_total = len(train_ds)
                n_train_eval = min(n_train_total, 5000)
                if n_train_eval < n_train_total:
                    rng_te = np.random.default_rng(args.seed)
                    eval_idx = rng_te.choice(n_train_total, n_train_eval,
                                             replace=False)
                    from torch.utils.data import Subset
                    train_eval_ds = Subset(train_ds, eval_idx.tolist())
                    print(f"[train-eval] Subsampling {n_train_eval:,} of "
                          f"{n_train_total:,} train patches for diagnostic "
                          f"scatter plot.")
                else:
                    train_eval_ds = train_ds

                train_eval_loader = DataLoader(train_eval_ds, batch_size=args.batch,
                                            shuffle=False, drop_last=False,
                                            num_workers=_max_workers,
                                            pin_memory=pin,
                                            persistent_workers=_max_workers > 0,
                                            prefetch_factor=(2 if _max_workers > 0
                                                             else None))

                model.eval()
                ys_tr, ps_tr, ms_tr = [], [], []
                with torch.no_grad():
                    for x, y, m in train_eval_loader:
                        x = x.to(device, non_blocking=True)
                        p = model(x)
                        ys_tr.append(y.numpy())
                        ps_tr.append(p.detach().cpu().numpy())
                        ms_tr.append(m.numpy())
                ys_tr = np.concatenate(ys_tr, axis=0)
                ps_tr = np.concatenate(ps_tr, axis=0)
                ms_tr = np.concatenate(ms_tr, axis=0).astype(bool)


                # Scaled-space metrics (comparable to [Test - Scaled Units] below)
                m_tr_mae, _, m_tr_rmse, m_tr_r2 = compute_metrics(ys_tr[ms_tr], ps_tr[ms_tr])
                print(f"[TRAIN - Scaled Units] MAE={m_tr_mae:.2e} "
                    f"RMSE={m_tr_rmse:.2e} R2={m_tr_r2:.3f}")

                metrics["train_mae_scaled"]  = float(m_tr_mae)
                metrics["train_rmse_scaled"] = float(m_tr_rmse)
                metrics["train_r2_scaled"]   = float(m_tr_r2)

                # Inverse transform for unscaled scatter
                if y_sklearn is not None:
                    ys_tr_plot = y_sklearn.inverse_transform(
                        ys_tr.reshape(-1, 1).astype(np.float64)
                    ).reshape(ys_tr.shape).astype(np.float32)
                    ps_tr_plot = y_sklearn.inverse_transform(
                        ps_tr.reshape(-1, 1).astype(np.float64)
                    ).reshape(ps_tr.shape).astype(np.float32)
                else:
                    ys_tr_plot = ys_tr
                    ps_tr_plot = ps_tr

                gt_tr = ys_tr_plot[ms_tr]
                pd_tr = ps_tr_plot[ms_tr]
                good = np.isfinite(gt_tr) & np.isfinite(pd_tr)
                gt_tr = gt_tr[good]
                pd_tr = pd_tr[good]

                if gt_tr.size > 0:
                    mae_u, _, rmse_u, r2_u = compute_metrics(gt_tr, pd_tr)
                    x_lo, x_hi = np.percentile(gt_tr, [1, 99])
                    y_lo, y_hi = np.percentile(pd_tr, [1, 99])
                    mn = min(x_lo, y_lo) * 0.9
                    mx = max(x_hi, y_hi) * 1.1
                    plt.figure()
                    hb = plt.hexbin(gt_tr, pd_tr, gridsize=150, cmap='inferno', mincnt=1, bins='log')
                    plt.colorbar(hb, label='Point Density (Log Scale)')
                    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1)
                    plt.grid(True, linestyle=':', alpha=0.5)
                    if (args.model == "multi_unet"
                            and getattr(args, "map_fields", None)):
                        xlabel = f"GT ({', '.join(args.map_fields)})"
                    else:
                        xlabel = f"GT ({args.map_field})"
                    plt.xlabel(xlabel)
                    plt.ylabel("Predicted")
                    plt.title(f"TRAIN Pred vs GT  N={gt_tr.size}  "
                            f"MAE={mae_u:.3e}  RMSE={rmse_u:.3e}  R2={r2_u:.3f}")
                    plt.xlim(mn, mx)
                    plt.ylim(mn, mx)
                    plt.gca().set_aspect('equal', 'box')
                    plt.tight_layout()
                    plt.savefig(os.path.join(results_dir, "train_scatter.png"), dpi=300)
                    plt.close()

                # Free train-eval buffers
                del train_eval_loader, ys_tr, ps_tr, ms_tr
                gc.collect()
            else:
                print("[BO] Skipping TRAIN evaluation + train_scatter.png.")


            # Test evaluation
            model.eval()
            ys_all, ps_all, ms_all = [], [], []
            with torch.no_grad():
                for x, y, m in test_loader:                # <-- 3 items
                    x = x.to(device, non_blocking=True)
                    p = model(x)
                    ys_all.append(y.numpy())
                    ps_all.append(p.detach().cpu().numpy())
                    ms_all.append(m.numpy())

            y_true_scaled = np.concatenate(ys_all, axis=0)
            y_pred_scaled = np.concatenate(ps_all, axis=0)
            m_test_all    = np.concatenate(ms_all, axis=0).astype(bool)

            # Metrics on masked pixels only (scaled units)
            y_masked = y_true_scaled[m_test_all]
            p_masked = y_pred_scaled[m_test_all]
            if y_masked.size > 0:
                mae_s, mse_s, rmse_s, r2_s = compute_metrics(y_masked, p_masked)
            else:
                mae_s, mse_s, rmse_s, r2_s = np.nan, np.nan, np.nan, np.nan
            print(f"[Test - Scaled Units] MAE = {mae_s:.2e} "
                  f"RMSE = {rmse_s:.2e} R2 = {r2_s:.3f}")

            metrics["test_mae_scaled"]  = float(mae_s)
            metrics["test_rmse_scaled"] = float(rmse_s)
            metrics["test_r2_scaled"]   = float(r2_s)

            # Free remaining training/test datasets and loaders before visualization
            try:
                del test_loader
            except NameError:
                pass
            if 'train_ds' in locals():
                del train_ds
            if 'test_ds' in locals():
                del test_ds
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ##### ---- VISUALIZATIONS ---- #####

        # Skip the entire visualization block if there are no test predictions
        if 'y_pred_scaled' not in locals():
            print("[viz] No test predictions available — skipping visualization.")
            _save_metrics_json(results_dir, metrics)
            return metrics

        # ---- BO trials: skip ALL per-sample test-set metrics and every
        # heavy visualization. The BO objective (val_r2 / val_blur_pen /
        # val_r2_minus_blur / val_r2_spectral / pareto_r2_blur) is
        # populated from the validation loop inside train_torch_model /
        # train_torch_model_gan and lives in metrics['val_*_best'].
        # Test quantities are explicitly forbidden as BO objectives by
        # _score_from_metrics(_from_bo=True), so computing them here is
        # both unnecessary and unsafe (map_data is deliberately dropped
        # during BO trials to save memory).
        if _is_bo:
            _save_metrics_json(results_dir, metrics)
            return metrics




        # ---- DIAGNOSTIC: scaled predictions ----
        print("=== y_pred_scaled stats (before inverse transform) ===")
        print(f"  shape  : {y_pred_scaled.shape}")
        print(f"  finite : {np.isfinite(y_pred_scaled).all()}")
        print(f"  NaN    : {int(np.isnan(y_pred_scaled).sum())}")
        print(f"  Inf    : {int(np.isinf(y_pred_scaled).sum())}")
        if np.isfinite(y_pred_scaled).any():
            fin = y_pred_scaled[np.isfinite(y_pred_scaled)]
            print(f"  finite min/max: {fin.min():.4f} / {fin.max():.4f}")

        # Convert scaled predictions back to real units
        if args.y_scaler != "none" and y_sklearn is not None:
            yp_shape = y_pred_scaled.shape
            y_pred_flat = y_pred_scaled.reshape(-1, 1).astype(np.float64)  # avoid fp32 overflow in expm1

            # For log-type scalers, expm1(large) overflows to +inf. Clip to a safe range.
            if args.y_scaler in ("log", "log_standard"):
                # log1p(1e8) ≈ 18.4 → clip above that to avoid overflow
                y_pred_flat = np.clip(y_pred_flat, -50.0, 50.0)

            finite_mask = np.isfinite(y_pred_flat).ravel()
            y_pred_out = np.full_like(y_pred_flat, np.nan)
            if finite_mask.any():
                inv = y_sklearn.inverse_transform(y_pred_flat[finite_mask])
                y_pred_out[finite_mask] = inv
            y_pred_test = y_pred_out.reshape(yp_shape).astype(np.float32)
        else:
            y_pred_test = y_pred_scaled

        # ---- DIAGNOSTIC: unscaled predictions ----
        print("=== y_pred_test stats (after inverse transform) ===")
        print(f"  finite : {np.isfinite(y_pred_test).all()}")
        print(f"  NaN    : {int(np.isnan(y_pred_test).sum())}")
        print(f"  Inf    : {int(np.isinf(y_pred_test).sum())}")
        if np.isfinite(y_pred_test).any():
            fin = y_pred_test[np.isfinite(y_pred_test)]
            print(f"  finite min/max: {fin.min():.4f} / {fin.max():.4f}")
        else:
            print("  WARNING: y_pred_test has no finite values — prediction plots will be empty")



        # ---- Reconstruct test prediction image(s) ---- #
        # Inverse transform test GT patches for visualization (same logic as predictions)
        if (args.out_avg_mode == "patch"
                and y_test_raw is not None
                and args.y_scaler != "none"
                and y_sklearn is not None):
            yt_shape = y_test_raw.shape
            yt_flat = y_test_raw.reshape(-1, 1).astype(np.float64)
            if args.y_scaler in ("log", "log_standard"):
                yt_flat = np.clip(yt_flat, -50.0, 50.0)
            finite = np.isfinite(yt_flat).ravel()
            yt_out = np.full_like(yt_flat, np.nan)
            if finite.any():
                yt_out[finite] = y_sklearn.inverse_transform(yt_flat[finite])
            y_test_unscaled = yt_out.reshape(yt_shape).astype(np.float32)
        else:
            y_test_unscaled = y_test_raw

        if args.target_mode == "map":
            # ===== MAP-MODE VISUALIZATION =====
            # For multi_unet, sample_map_all has shape (H, W, n_maps) and
            # sample_preds_all has shape (n, P, P, n_maps). We loop over each
            # map channel and produce one set of visualizations per map.
            is_multi = (args.model == "multi_unet"
                        and getattr(args, "map_fields", None))
            map_names = (list(args.map_fields) if is_multi
                         else [args.map_field])

            patch_idx = 0
            for meta in meta_test:
                n = meta["n_patches"]
                sample_name = meta["name"]
                sample_map_all = meta["map_data"]   # (H, W) or (H, W, n_maps)
                sample_mask = meta["mask"]
                sample_coords = meta["coords"]
                sample_preds_all = y_pred_test[patch_idx : patch_idx + n]
                patch_idx += n

                for ch_idx, ch_name in enumerate(map_names):
                    if is_multi:
                        sample_map = sample_map_all[..., ch_idx]
                        sample_preds = sample_preds_all[..., ch_idx]
                        ch_tag = f"{sample_name}__{ch_name}"
                    else:
                        sample_map = sample_map_all
                        sample_preds = sample_preds_all
                        ch_tag = sample_name

                    H_s, W_s = sample_map.shape

                    if args.out_avg_mode != "patch":
                        pred_img = np.full((H_s, W_s), np.nan, dtype=np.float32)
                        for (r, c), v in zip(sample_coords, sample_preds):
                            pred_img[r, c] = v
                    else:
                        P_out = args.out_patch
                        rad_out = P_out // 2
                        yy, xx = np.mgrid[0:P_out, 0:P_out]
                        sigma = max(P_out / 4.0, 1.0)
                        gw = np.exp(
                            -((yy - P_out // 2) ** 2 + (xx - P_out // 2) ** 2)
                            / (2.0 * sigma ** 2)
                        ).astype(np.float32)
                        pred_sum = np.zeros((H_s, W_s), dtype=np.float32)
                        pred_count = np.zeros((H_s, W_s), dtype=np.float32)
                        for (r, c), patch_pred in zip(sample_coords, sample_preds):
                            r0, r1 = r - rad_out, r + rad_out + 1
                            c0, c1 = c - rad_out, c + rad_out + 1
                            if r0 < 0 or c0 < 0 or r1 > H_s or c1 > W_s:
                                # Patch is partially outside the image grid.
                                # Skip rather than silently misalign.
                                continue
                            pred_sum[r0:r1, c0:c1] += patch_pred * gw
                            pred_count[r0:r1, c0:c1] += gw

                        pred_img = np.full((H_s, W_s), np.nan, dtype=np.float32)
                        valid = pred_count > 0
                        pred_img[valid] = pred_sum[valid] / pred_count[valid]

                    gt_vals_vis = sample_map[sample_mask]
                    if gt_vals_vis.size > 0:
                        vmin = np.percentile(gt_vals_vis, 1)
                        vmax = np.percentile(gt_vals_vis, 99)
                    else:
                        vmin, vmax = None, None

                    save_path = os.path.join(results_dir,
                                             f"test_map_{ch_tag}.png")

                    plot_maps(
                        sample_map,
                        pred_img,
                        title=ch_tag,
                        subtitle=f"{ch_name}: GT vs prediction",
                        mask=sample_mask.astype(bool),
                        save_path=save_path,
                    )

                    plt.close()
                    valid_px = (sample_mask.astype(bool)
                                & np.isfinite(sample_map)
                                & np.isfinite(pred_img))
                    gt_vals = sample_map[valid_px]
                    pred_vals = pred_img[valid_px]
                    if gt_vals.size > 0:
                        mae, mse, rmse, r2 = compute_metrics(gt_vals, pred_vals)
                        metrics["test_mae_unscaled_per_sample"][ch_tag]  = float(mae)
                        metrics["test_rmse_unscaled_per_sample"][ch_tag] = float(rmse)
                        metrics["test_r2_unscaled_per_sample"][ch_tag]   = float(r2)

                        try:
                            ratio = local_std_ratio(
                                gt=sample_map,
                                pred=pred_img,
                                mask=sample_mask.astype(bool),
                                k=5,
                            )
                            metrics["test_local_std_ratio_per_sample"][ch_tag] = (
                                float(ratio) if np.isfinite(ratio) else None
                            )
                        except Exception as e:
                            print(f"[local_std_ratio] failed for {ch_tag}: {e}")
                            metrics["test_local_std_ratio_per_sample"][ch_tag] = None

                        x_lo, x_hi = np.percentile(gt_vals, [1, 99])
                        y_lo, y_hi = np.percentile(pred_vals, [1, 99])
                        mn = min(x_lo, y_lo) * 0.9
                        mx = max(x_hi, y_hi) * 1.1
                        plt.figure()
                        hb = plt.hexbin(gt_vals, pred_vals, gridsize=150,
                                        cmap='inferno', mincnt=1, bins='log')
                        plt.colorbar(hb, label='Point Density (Log Scale)')
                        plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1)
                        plt.grid(True, linestyle=':', alpha=0.5)
                        plt.xlabel(f"Ground truth ({ch_name})")
                        plt.ylabel("Predicted")
                        plt.title(f"{ch_tag}  N={gt_vals.size}  "
                                  f"MAE={mae:.3e}  RMSE={rmse:.3e}  R2={r2:.3f}")
                        plt.xlim(mn, mx); plt.ylim(mn, mx)
                        plt.gca().set_aspect('equal', 'box')
                        plt.tight_layout()
                        plt.savefig(os.path.join(results_dir,
                                    f"scatter_plot_{ch_tag}.png"), dpi=300)
                        plt.close()

                    extract_and_save_profiles(
                        gt=sample_map, pred=pred_img,
                        mask=sample_mask.astype(bool),
                        sample_name=ch_tag, out_dir=results_dir,
                    )

                    if (args.out_avg_mode == "patch"
                            and y_test_unscaled is not None
                            and m_test_raw is not None):
                        start = patch_idx - n
                        sample_gt_patches_all   = y_test_unscaled[start:patch_idx]
                        sample_pred_patches_all = y_pred_test[start:patch_idx]
                        sample_mask_patches     = m_test_raw[start:patch_idx]
                        if is_multi:
                            sample_gt_patches   = sample_gt_patches_all[..., ch_idx]
                            sample_pred_patches = sample_pred_patches_all[..., ch_idx]
                        else:
                            sample_gt_patches   = sample_gt_patches_all
                            sample_pred_patches = sample_pred_patches_all
                        plot_random_tiles(
                            y_true=sample_gt_patches,
                            y_pred=sample_pred_patches,
                            m_out=sample_mask_patches,
                            sample_name=ch_tag, out_dir=results_dir,
                            n_tiles=3, seed=args.seed, show_diff=True,
                        )


        else:
            # ===== SPECTRA-MODE VISUALIZATION =====
            patch_idx = 0
            for meta in meta_test:
                n = meta["n_patches"]
                sample_name = meta["name"]
                sample_coords = meta["coords"]
                sample_preds = y_pred_test[patch_idx : patch_idx + n]
                patch_idx += n

                ftir_cube = meta["map_data"]
                ftir_wls_s = meta.get("ftir_wls", None)
                mask_s    = meta.get("mask", None)            # <-- define here, once
                if (ftir_cube is None or ftir_cube.ndim != 3
                        or ftir_wls_s is None or mask_s is None):
                    print(f"[viz spectra] Missing FTIR cube/wls/mask for {sample_name}; skipping.")
                    continue

                # Apply same output derivatives to GT for fair comparison
                if args.out_der_mode != "raw":
                    Hf, Wf, Cf = ftir_cube.shape
                    ftir_vis = apply_derivatives_simple(
                        ftir_cube.reshape(-1, Cf),
                        der_mode=args.out_der_mode,
                        ftir_wls=ftir_wls_s,
                    ).reshape(Hf, Wf, -1)
                    if args.out_der_mode == "raw+der1":
                        vis_wls = np.concatenate([ftir_wls_s, ftir_wls_s])
                    elif args.out_der_mode == "raw+der2":
                        vis_wls = np.concatenate([ftir_wls_s, ftir_wls_s])
                    elif args.out_der_mode == "raw+der1+der2":
                        vis_wls = np.concatenate([ftir_wls_s, ftir_wls_s, ftir_wls_s])
                    elif args.out_der_mode == "der1+der2":
                        vis_wls = np.concatenate([ftir_wls_s, ftir_wls_s])
                    else:  # der1 / der2
                        vis_wls = ftir_wls_s
                else:
                    ftir_vis = ftir_cube
                    vis_wls = ftir_wls_s

                C_ftir_vis = ftir_vis.shape[2]

                pred_cube = build_pred_cube_from_outputs(
                    ftir_cube=ftir_vis,
                    coords_test=sample_coords,
                    y_pred_test=sample_preds,
                    C_ftir=C_ftir_vis,
                    out_patch=args.out_patch,
                )

                if pred_cube is not None:
                    # ---- Per-sample unscaled metrics (needed for BO scoring) ----
                    mask_s_bool = mask_s.astype(bool)
                    mask3d = np.broadcast_to(mask_s_bool[..., None], ftir_vis.shape)
                    valid = mask3d & np.isfinite(ftir_vis) & np.isfinite(pred_cube)
                    gt_v = ftir_vis[valid]
                    pr_v = pred_cube[valid]
                    if gt_v.size > 0:
                        mae, _, rmse, r2 = compute_metrics(gt_v, pr_v)
                        metrics["test_mae_unscaled_per_sample"][sample_name]  = float(mae)
                        metrics["test_rmse_unscaled_per_sample"][sample_name] = float(rmse)
                        metrics["test_r2_unscaled_per_sample"][sample_name]   = float(r2)
                        print(f"[Test - Unscaled, {sample_name}] "
                            f"MAE={mae:.3e} RMSE={rmse:.3e} R2={r2:.3f}")
                        # Decompose R^2 into spectral-shape vs spatial-fidelity
                        # components for spectra-mode reporting. The flattened
                        # r2 above conflates inter-pixel mean differences with
                        # intra-pixel spectral shape fidelity. The decomposition
                        # separates them; see r2_components_spectra() docstring.
                        # ftir_vis: (H, W, C_ftir), pred_cube: (H, W, C_ftir).
                        # Build (N_pixels, C_ftir) matrices over the mask only.
                        try:
                            mask_2d = mask_s_bool & np.all(
                                np.isfinite(ftir_vis), axis=-1
                            ) & np.all(np.isfinite(pred_cube), axis=-1)
                            if mask_2d.any():
                                yt_2d = ftir_vis[mask_2d]    # (N_valid, C_ftir)
                                yp_2d = pred_cube[mask_2d]   # (N_valid, C_ftir)
                                r2_dec = r2_components_spectra(
                                    yt_2d, yp_2d, mask_2d.ravel()
                                )
                                metrics.setdefault(
                                    "test_r2_global_per_sample", {}
                                )[sample_name] = float(r2_dec["r2_global"])
                                metrics.setdefault(
                                    "test_r2_spectral_per_sample", {}
                                )[sample_name] = float(r2_dec["r2_spectral"])
                                metrics.setdefault(
                                    "test_r2_spatial_per_sample", {}
                                )[sample_name] = float(r2_dec["r2_spatial"])
                                print(
                                    f"[Test - R2 decomposition, "
                                    f"{sample_name}] "
                                    f"global={r2_dec['r2_global']:.3f}  "
                                    f"spectral(per-pixel)={r2_dec['r2_spectral']:.3f}  "
                                    f"spatial(per-channel)={r2_dec['r2_spatial']:.3f}"
                                )
                        except Exception as e:
                            print(f"[r2_components_spectra] failed for "
                                  f"{sample_name}: {e}")

                        # 2D summary for blur penalty (mean across spectral channels)
                        try:
                            with np.errstate(invalid="ignore"):
                                gt2d = np.nanmean(ftir_vis,  axis=-1)
                                pr2d = np.nanmean(pred_cube, axis=-1)
                            ratio = local_std_ratio(gt2d, pr2d, mask_s_bool, k=5)
                            metrics["test_local_std_ratio_per_sample"][sample_name] = (
                                float(ratio) if np.isfinite(ratio) else None
                            )
                        except Exception as e:
                            print(f"[local_std_ratio spectra] failed for {sample_name}: {e}")
                            metrics["test_local_std_ratio_per_sample"][sample_name] = None

                        # Per-sample scatter plot (parity with map mode)
                        x_lo, x_hi = np.percentile(gt_v, [1, 99])
                        y_lo, y_hi = np.percentile(pr_v, [1, 99])
                        mn = min(x_lo, y_lo) * 0.9
                        mx = max(x_hi, y_hi) * 1.1
                        plt.figure()
                        hb = plt.hexbin(gt_v, pr_v, gridsize=150, cmap="inferno", mincnt=1, bins='log')
                        plt.colorbar(hb, label="Point Density (Log Scale)")
                        plt.plot([mn, mx], [mn, mx], "r--", linewidth=1)
                        plt.grid(True, linestyle=':', alpha=0.5)
                        plt.xlabel("FTIR (GT)")
                        plt.ylabel("Predicted")
                        plt.title(f"{sample_name}  N={gt_v.size}  "
                                f"MAE={mae:.3e}  RMSE={rmse:.3e}  R2={r2:.3f}")
                        plt.xlim(mn, mx); plt.ylim(mn, mx)
                        plt.gca().set_aspect("equal", "box")
                        plt.tight_layout()
                        plt.savefig(os.path.join(results_dir,
                                                f"scatter_plot_{sample_name}.png"), dpi=300)
                        plt.close()

                    test_cols_full = (0, ftir_vis.shape[1])
                    try:
                        show_slider(ftir_vis, pred_cube, vis_wls, test_cols_full,
                                    cmap="viridis", plot_range="percentile",
                                    error_mode="signed")
                    except Exception as e:
                        print(f"[viz spectra] show_slider failed for {sample_name}: {e}")


                # Per-sample point spectra (mask_s already defined above)
                masked_coords = [(r, c) for r, c in sample_coords if mask_s[r, c]]
                if masked_coords:
                    n_ex = min(5, len(masked_coords))
                    rng = np.random.default_rng(args.seed)
                    ex_idx = rng.choice(len(masked_coords), size=n_ex, replace=False)
                    fig_path = os.path.join(results_dir,
                                            f"spectra_{sample_name}.png")
                    plot_spectra_at_points(
                        list(ex_idx), ftir_vis, sample_coords, sample_preds,
                        ftir_wls_s[:C_ftir_vis], save_path=fig_path,
                    )

                # Save predictions + raw GT cube for this sample.
                # Storing the raw FTIR cube (already downsampled to match the prediction
                # grid) allows for future re-processing with different derivative options
                sample_mask_save = (
                    meta["mask"].astype(np.bool_) if meta.get("mask") is not None else None
                )

                npz_path = os.path.join(results_dir,
                                        f"test_visuals_{sample_name}.npz")
                np.savez_compressed(
                    npz_path,
                    ftir_wls=ftir_wls_s,                                  # (C_ftir,)
                    coords_test=np.array(sample_coords, dtype=np.int32),  # (N, 2)
                    y_pred_test=sample_preds,                             # already inverse-scaled
                    ftir_cube=ftir_cube.astype(np.float32),               # (H, W, C_ftir) raw
                    cartilage_mask=(sample_mask_save
                                    if sample_mask_save is not None
                                    else np.array([], dtype=np.bool_)),
                    out_der_mode=str(args.out_der_mode),                  # for sanity-check
                    out_patch=int(args.out_patch),
                )
                print(f"[viz spectra] Saved {npz_path} "
                    f"(ftir_cube={ftir_cube.shape}, preds={sample_preds.shape})")




        # ---- Aggregate unscaled test R2 across all test samples ----
        r2_dict = metrics.get("test_r2_unscaled_per_sample", {})
        if r2_dict:
            vals = [v for v in r2_dict.values()
                    if v is not None and np.isfinite(v)]
            if vals:
                r2_mean = float(np.mean(vals))
                r2_std  = float(np.std(vals))
                print(f"[Test - Unscaled Units] "
                      f"per-sample R2 mean={r2_mean:.3f}  "
                      f"std={r2_std:.3f}  (n_samples={len(vals)})")
                for name in sorted(r2_dict):
                    v = r2_dict[name]
                    if v is not None and np.isfinite(v):
                        print(f"    {name}: R2={v:.3f}")

        # ---- Persist metrics for later comparison ----
        _save_metrics_json(results_dir, metrics)


        # ---- Update registry row with metrics (skip for BO trials) ----
        if not _is_bo:
            update_row = {
                "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
                "short_tag":   short_tag,
                "fold_tag":    fold_tag,
                "results_dir": results_dir,
                "run_tag":     run_tag,
            }
            update_row.update(_args_to_excel_row(args))
            update_row.update(_flatten_for_excel(metrics, parent_key="metric"))
            append_run_to_excel(registry_path, update_row)




        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            
        return metrics

    finally:
        # ---- Section timing summary ----
        try:
            total_dt = time.time() - _fold_t0
            print(f"\n[time] === Section summary for {fold_tag} "
                  f"(total {total_dt:.1f}s) ===", flush=True)
            for k, v in sorted(_section_times.items(),
                               key=lambda kv: -kv[1]):
                print(f"[time]   {k:40s} {v:8.1f}s "
                      f"({100*v/max(total_dt,1e-9):5.1f}%)", flush=True)
        except Exception:
            pass

        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        if _log_fh is not None:
            try:
                _log_fh.close()
            except Exception:
                pass



# ---- Build splits: LOSO or fixed prefix-based ----
def _build_splits_loso(samples, test_idx, val_strategy="next"):
    test_samples = [samples[test_idx]]
    val_idx = None
    if val_strategy in ("next", "prev") and len(samples) > 1:
        if val_strategy == "next":
            val_idx = (test_idx + 1) % len(samples)
        else:
            val_idx = (test_idx - 1) % len(samples)
        if val_idx == test_idx:
            val_idx = None
    val_samples = [samples[val_idx]] if val_idx is not None else []
    train_samples = [
        s for i, s in enumerate(samples)
        if i != test_idx and i != val_idx
    ]
    return train_samples, val_samples, test_samples


def _build_splits_prefix(samples, test_prefix, val_prefix):
    tp, vp = test_prefix.upper(), val_prefix.upper()
    train_samples, val_samples, test_samples = [], [], []
    for s in samples:
        name = s["name"].upper()
        if name.startswith(tp):
            test_samples.append(s)
        elif name.startswith(vp):
            val_samples.append(s)
        else:
            train_samples.append(s)
    return train_samples, val_samples, test_samples


# ----------------- Main ----------------- #

def main():
    args = parse_args()

    if args.bo_metric == "auto":
        args.bo_metric = ("val_r2"
                        if args.target_mode == "map"
                        else "val_r2_spectral")

    # Backwards compatibility: the legacy label 'cyclegan' is remapped
    # to 'pix2pix_inv', which is the accurate name for the paired
    # inverse-consistency loss used in this work (see Methods).
    if getattr(args, "gan_mode", "none") == "cyclegan":
        print("[gan_mode] Legacy value 'cyclegan' remapped to "
              "'pix2pix_inv' (paired inverse-consistency reconstruction). "
              "See Methods for the definition.")
        args.gan_mode = "pix2pix_inv"



    if args.results_dir is None:
        args.results_dir = (r"F:\results_maps" if args.target_mode == "map"
                            else r"F:\results_spectra")

    run_tag = make_run_tag(args)
    print(f"Run tag: {run_tag}\n")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    # Expose seed to module-level worker init (picklable across processes).
    global _WORKER_BASE_SEED
    _WORKER_BASE_SEED = int(args.seed)
    args._worker_seed_init = _worker_seed_init

    if args.model in ("unet", "unet_simple", "pretrained_unet"):
        if args.avg_mode != "patch":
            raise ValueError(f"For model='{args.model}', please use --avg_mode patch so input is (P,P,C).")
        if args.out_avg_mode not in ["avg", "flat", "patch"]:
            raise ValueError(f"For model='{args.model}', out_avg_mode must be one of: avg, flat, patch.")

    # ---- Reject unsupported model + target combinations ----
    if args.target_mode == "spectra" and args.model in {"cnn", "unet_simple"}:
        raise ValueError(
            f"Model '{args.model}' is not supported in target_mode=spectra. "
            f"Supported spectra models: 'unet' (UNet2D_FTIR), 'mlp' (SpectrumMLP), "
            f"'cnn1d' (SpectralCNN1D), 'plsr' (PLSRWrapperSpectra)."
        )
    if args.target_mode == "spectra" and args.model in {"mlp", "cnn1d"} \
            and (args.patch != 1 or args.avg_mode != "avg"):
        raise ValueError(
            f"target_mode=spectra with model='{args.model}' requires "
            f"--patch 1 --avg_mode avg (per-pixel spectrum input)."
        )

    if (args.target_mode == "spectra"
            and args.out_der_mode != "raw"
            and args.out_avg_mode == "flat"):
        raise ValueError(
            "Combination target_mode=spectra + out_avg_mode=flat + "
            "out_der_mode!=raw is not supported. Use out_avg_mode in "
            "{avg, patch} when applying output derivatives."
        )



    # Block invalid PLSR config (PLSR currently supports only scalar output)
    if (args.model == "plsr"
            and args.out_avg_mode != "avg"
            and args.target_mode == "map"):
        raise ValueError(
            "PLSR (map mode) currently supports only --out_avg_mode avg."
        )


    if args.patch_dir is not None and args.spatial_scale < 1.0:
        print("NOTE: --spatial_scale is ignored when --patch_dir is set. "
            "Downsampling must be applied during patch extraction.")


    # ---- Discover samples and assign to splits ----
    if args.patch_dir is not None:
        # Discover samples from .npz filenames — no raw data needed
        samples = discover_samples_from_npz(args.patch_dir)
        if not samples:
            raise FileNotFoundError(
                f"No .npz patch files found in {args.patch_dir}"
            )
    else:
        # Discover from raw .mat files
        if args.data_dir is None:
            raise ValueError("--data_dir is required when --patch_dir is not set.")
        samples = discover_sample_files(args.data_dir, target_mode=args.target_mode)
        if not samples:
            raise FileNotFoundError(
                f"No matched ftir_maps/uhsi file pairs found in {args.data_dir}"
            )

    print(f"Found {len(samples)} sample(s):")
    for s in samples:
        print(f"  {s['name']}")
    print()


    # Build the list of folds (each entry is one full training run)
    if args.loso:
        if args.loso_only_samples:
            wanted = {n.lower() for n in args.loso_only_samples}
            test_indices = [i for i, s in enumerate(samples)
                            if s["name"].lower() in wanted]
            if not test_indices:
                raise ValueError(
                    f"No samples match --loso_only_samples="
                    f"{args.loso_only_samples}"
                )
        else:
            test_indices = list(range(len(samples)))


        # METHODOLOGICAL CONTRACT: if BO was run with a fixed prefix split,
        # exclude the BO-test and BO-val samples from the LOSO test rotation,
        # because hyperparameters were chosen with those samples in the loop.
        # This prevents meta-level test-set leakage.
        if getattr(args, "_bo_was_run", False) and args.bo_proxy == "single_fold":
            tp = args.test_prefix.upper()
            vp = args.val_prefix.upper()
            excluded = [i for i, s in enumerate(samples)
                        if s["name"].upper().startswith(tp)
                        or s["name"].upper().startswith(vp)]
            if excluded:
                print(f"[LOSO] Excluding BO-test/val samples from LOSO test "
                    f"rotation (meta-leakage prevention): "
                    f"{[samples[i]['name'] for i in excluded]}")
                test_indices = [i for i in test_indices if i not in excluded]


        folds = []
        for i in test_indices:
            tr, va, te = _build_splits_loso(
                samples, i, args.loso_val_strategy
            )
            fold_tag = f"loso_{samples[i]['name']}"
            folds.append((fold_tag, tr, va, te))

        print(f"\n=== LOSO mode: {len(folds)} fold(s) ===")
    else:
        tr, va, te = _build_splits_prefix(
            samples, args.test_prefix, args.val_prefix
        )
        folds = [("fixed", tr, va, te)]


    # ---- Determine base output directory ----
    if args.results_dir is not None:
        results_base = args.results_dir
    elif args.patch_dir is not None:
        results_base = os.path.join(args.patch_dir, "mve")
    else:
        results_base = os.path.join(args.data_dir, "mve")

    os.makedirs(results_base, exist_ok=True)

    if args.bo:
        if args.bo_storage is None:
            args.bo_storage = f"sqlite:///{os.path.join(results_base, 'bo.db')}"
            args.bo_storage = args.bo_storage.replace("\\", "/")

        from optuna.storages import RDBStorage
        args._bo_storage_obj = RDBStorage(
            url=args.bo_storage,
            engine_kwargs={"connect_args": {"timeout": 30}},
            heartbeat_interval=60,
            grace_period=120,
            failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=3),
        )

    user_epochs = args.epochs

    # ---- Run BO if requested, else run normal pipeline ----
    if args.bo:
        # Methodological contract:
        #   - BO reads only VALIDATION metrics.
        #   - The TEST sample (test_prefix) is held out for final
        #     reporting in the post-BO refit and for LOSO aggregation.
        assert args.bo_metric in (
            "val_r2", "val_loss",
            "val_r2_minus_blur", "val_r2_spectral",
            "pareto_r2_blur",
        ), (
            f"bo_metric={args.bo_metric!r} is not allowed. Use a "
            f"validation-only objective: val_r2, val_loss, "
            f"val_r2_minus_blur, val_r2_spectral (spectra mode only), "
            f"or pareto_r2_blur (multi-objective NSGA-II)."
        )

        print("=== Running Bayesian hyperparameter optimization ===")
        print(f"  Protocol:")
        print(f"    BO objective       : {args.bo_metric}")
        print(f"    BO proxy           : {args.bo_proxy}")
        print(f"    Train (in BO)      : everything except "
              f"{args.test_prefix}_* and {args.val_prefix}_*")
        print(f"    Validation (in BO) : {args.val_prefix}_*")
        print(f"    Test (NEVER in BO) : {args.test_prefix}_*")
        if args.loso:
            print(f"    Final evaluation   : LOSO across all samples")
        study = run_bo(args, samples, results_base)
        args._bo_was_run = True
        # Release the SQLite handle so the post-BO refit doesn't hold a stale
        # database connection (matters when results_dir is on networked storage).
        args._bo_storage_obj = None

        if args.bo_metric == "pareto_r2_blur":
            knee = getattr(study, "_uhsi_knee_trial", None)
            if knee is None:
                print("[BO] Pareto study returned no usable knee point; "
                    "keeping current args and skipping refit.")
                best = {}
            else:
                print(f"[BO] Applying knee-point params from trial "
                    f"#{knee.number} "
                    f"(val_r2={knee.values[0]:.4f}, "
                    f"blur={knee.values[1]:.4f}).")
                best = dict(knee.params)
        else:
            try:
                best = study.best_params
            except ValueError:
                print("[BO] No completed trial — keeping current args and "
                    "skipping re-run.")
                best = {}

        for k, v in best.items():
            setattr(args, k, v)
        args.epochs = user_epochs

        print(f"\n=== Re-running full pipeline with best params ===")
        args.bo = False  # prevent recursion if you reuse main()

    # ---- Run each fold (existing code) ----
    all_folds_metrics = []
    for fold_tag, train_samples, val_samples, test_samples in folds:
        print(f"\n{'='*70}")
        print(f"FOLD: {fold_tag}")
        print(f"  Train: {[s['name'] for s in train_samples]}")
        print(f"  Val:   {[s['name'] for s in val_samples]}")
        print(f"  Test:  {[s['name'] for s in test_samples]}")
        print(f"{'='*70}")

        fold_metrics = run_one_fold(
            args=args,
            samples=samples,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            fold_tag=fold_tag,
            results_base=results_base,
        )
        all_folds_metrics.append({"fold": fold_tag, "metrics": fold_metrics})


    # ---- LOSO summary ----
    if args.loso:
        run_tag = make_run_tag(args)
        short_tag = make_short_tag(args, run_tag)
        summary_dir = os.path.join(results_base, short_tag)
        os.makedirs(summary_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(summary_dir, f"loso_summary_{ts}.json")
        with open(summary_path, "w") as f:
            json.dump(all_folds_metrics, f, indent=2, default=str)
        print(f"\nLOSO summary saved to: {summary_path}")

        # Aggregate per-sample R2/MAE
        test_r2 = [
            v for f in all_folds_metrics
            for v in f["metrics"].get("test_r2_unscaled_per_sample", {}).values()
            if v is not None and np.isfinite(v)
        ]
        test_mae = [
            v for f in all_folds_metrics
            for v in f["metrics"].get("test_mae_unscaled_per_sample", {}).values()
            if v is not None and np.isfinite(v)
        ]

        # Diagnostic: how many samples were dropped due to undefined R^2
        n_total_samples = sum(
            len(f["metrics"].get("test_r2_unscaled_per_sample", {}))
            for f in all_folds_metrics
        )
        n_finite_samples = len(test_r2)
        if n_finite_samples < n_total_samples:
            print(f"\nWARNING: {n_total_samples - n_finite_samples} of "
                f"{n_total_samples} sample-map pairs had undefined R^2 "
                f"(constant ground truth or too few pixels) and were "
                f"excluded from the mean. This can bias the reported "
                f"mean upward.")


        if test_r2:
            print(f"\nLOSO aggregated (all samples × all maps): "
                f"R2 mean={np.mean(test_r2):.3f} "
                f"std={np.std(test_r2):.3f}  "
                f"MAE mean={np.mean(test_mae):.3e}  "
                f"n_observations={len(test_r2)}")

            # Non-parametric 95% CI on the mean per-sample R^2, obtained
            # by cluster bootstrap resampling at the sample level (not
            # at the sample-map pair level). Clustering by sample
            # respects the dependence structure introduced by
            # --model multi_unet, which yields multiple R^2 values from
            # the same biological specimen. Per-sample R^2 is generally
            # not Gaussian, so a percentile bootstrap is preferred to a
            # symmetric mean ± std interval.

            sample_r2 = {}
            for f in all_folds_metrics:
                d = f["metrics"].get("test_r2_unscaled_per_sample", {})
                for tag, v in d.items():
                    if v is None or not np.isfinite(v):
                        continue
                    sample_name = tag.split("__", 1)[0]
                    sample_r2.setdefault(sample_name, []).append(float(v))

            n_samples = len(sample_r2)
            if n_samples >= 5:
                rng_boot = np.random.default_rng(args.seed)
                n_boot = 10_000
                sample_names = list(sample_r2.keys())
                boot_means = np.empty(n_boot, dtype=np.float64)
                for b in range(n_boot):
                    idx = rng_boot.integers(0, n_samples, n_samples)
                    vals = []
                    for j in idx:
                        vals.extend(sample_r2[sample_names[j]])
                    boot_means[b] = float(np.mean(vals))
                ci_lo = float(np.percentile(boot_means, 2.5))
                ci_hi = float(np.percentile(boot_means, 97.5))
                print(f"LOSO cluster bootstrap 95% CI on mean R^2 "
                      f"(resampling by sample): "
                      f"[{ci_lo:.3f}, {ci_hi:.3f}]  "
                      f"(n_samples={n_samples}, n_boot={n_boot})")
            else:
                print(f"LOSO cluster bootstrap CI skipped "
                      f"(n_samples={n_samples} < 5).")


            # Per-map breakdown when --model multi_unet (tag = "sample__map").
            per_map = {}
            for f in all_folds_metrics:
                d = f["metrics"].get("test_r2_unscaled_per_sample", {})
                for tag, v in d.items():
                    if v is None or not np.isfinite(v):
                        continue
                    if "__" in tag:
                        _, mp = tag.rsplit("__", 1)
                    else:
                        mp = args.map_field
                    per_map.setdefault(mp, []).append(float(v))
            if per_map:
                print("\nLOSO per-map R^2 mean ± std:")
                for mp in sorted(per_map):
                    vals = per_map[mp]
                    print(f"  {mp:>26}: "
                          f"mean={np.mean(vals):.3f}  "
                          f"std={np.std(vals):.3f}  "
                          f"n={len(vals)}")

    plt.show()


    # # ---- DEBUG / QUICK TEST LIMITS ----
    # train_samples = train_samples[:2]
    # val_samples = val_samples[:1]
    # test_samples = test_samples[:1]



if __name__ == "__main__":
    main()