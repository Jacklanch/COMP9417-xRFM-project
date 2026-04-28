# COMP9417 Group Project — xRFM Feature Learning Kernel Machines

Code archive for our report on xRFM: feature-learning kernel machines with an adaptive tree structure for tabular data.

**+bonus** — this submission includes the residual-weighted AGOP bonus section.

## Group Members

- [Name 1], z[zID]
- [Name 2], z[zID]
- [Name 3], z[zID]
- [Name 4], z[zID]
- [Name 5], z[zID]

## What This Project Does

Trains three models (xRFM, XGBoost, Random Forest) on five tabular datasets and produces:
1. A full comparison table of test-set performance, training time, and inference time.
2. A scaling experiment showing how each model's performance and training time change with training set size.
3. An interpretability comparison between xRFM's AGOP, PCA loadings, mutual information, and permutation importance.
4. A bonus extension: the residual-weighted AGOP, with a disagreement example and a performance comparison.

## Environment Setup

### Requirements
- Python 3.11
- CUDA-capable GPU (tested on RTX 4060 with CUDA 12.5 driver)
- ~80 GB free disk space (for Miniconda + PyTorch + datasets)

### Install (from scratch)

```bash
# 1. Install Miniconda (https://docs.conda.io/en/latest/miniconda.html)

# 2. Create and activate a fresh environment
conda create -n xrfm_project python=3.11 -y
conda activate xrfm_project

# 3. Install PyTorch with CUDA 12.1 (backward-compatible with CUDA 12.5 driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install xRFM with CUDA extras
pip install "xrfm[cu12]"

# 5. Install remaining libraries
pip install xgboost scikit-learn pandas numpy matplotlib seaborn jupyter ipykernel ucimlrepo scipy

# 6. Verify
python -c "import torch; from xrfm import xRFM; print('OK — CUDA:', torch.cuda.is_available())"
```

## How to Run

Open a terminal in the project folder (with the env activated) and start Jupyter:

```bash
jupyter notebook
```

Then execute the notebooks **in order**, top-to-bottom, running every cell:

1. `01_data_loading.ipynb` — downloads and preprocesses 5 datasets via the `ucimlrepo` package. First run takes ~3-5 minutes (UCI fetches each dataset). Subsequent runs are instant (datasets cached). Produces `processed/*.pkl`.

2. `02_model_training.ipynb` — trains xRFM, XGBoost, Random Forest on all 5 datasets. **Takes ~20-40 minutes** on GPU. Produces `models/*.pkl` and `results/tables/model_results.csv`.

3. `03_scaling_experiment.ipynb` — scaling experiment on Online Shoppers Intent at 5 training sizes. **Takes ~5-10 minutes.** Produces scaling CSV and plots.

4. `04_interpretability.ipynb` — interpretability comparison on Concrete Compressive Strength. Takes ~2 minutes. Produces heatmap, bars, and correlation plots.

5. `05_bonus_residual_agop.ipynb` — residual-weighted AGOP bonus. Takes ~3 minutes. Produces bonus plots and tables.

### Reproducibility

All random seeds are fixed at `RANDOM_SEED = 42`. Running any notebook top-to-bottom in a fresh kernel should produce numerically identical results to those in the report (within floating-point tolerance for GPU operations, which can introduce minor non-determinism in CUDA kernels).

## Datasets

All datasets are loaded programmatically from the UCI Machine Learning Repository via the `ucimlrepo` Python package — no manual download required. None of the chosen datasets overlap with the experiments showcased in the xRFM paper (Beaglehole et al., 2025).

| Dataset | Source | Task | Purpose |
|---|---|---|---|
| Concrete Compressive Strength | UCI (id=165) | Regression | Materials science, simple feature space, used for interpretability analysis |
| Energy Efficiency | UCI (id=242) | Regression | Building heating-load prediction |
| Bike Sharing | UCI (id=275) | Regression | Mixed types, n > 10k, d > 50 (after one-hot encoding) |
| Online Shoppers Intent | UCI (id=468) | Classification (binary) | Mixed types, n > 10k, used for scaling experiment |
| Statlog Shuttle | UCI (id=148) | Classification (multi-class, 7) | NASA telemetry, large (n=58k) |

## Known Issues (documented in the report)

- **xRFM scaling on Online Shoppers:** xRFM training failed to complete at n_train >= 5,000 on this dataset. Successful runs at smaller sizes (n=500, 1500, 3000) showed AUC stuck at 0.67-0.75 vs XGBoost ≈ 0.92. Discussed in the report as a finding about xRFM's reliability on imbalanced binary classification.
- **Shuttle multi-class AUC:** xRFM's AUC (0.612) was substantially below XGBoost (0.981) despite reasonable accuracy (0.787). Likely a probability calibration issue with multi-class imbalanced data.

## Results Files (for Report Authors)

All final numbers and figures are in:
- `results/tables/` — CSV files, including `model_results.csv` (master results) and the pivot tables per metric.
- `results/figures/` — PNG plots suitable for inclusion in the report.
- `REPORT_READY_SUMMARY.md` — consolidated plain-English summary of every experiment with key numbers and suggested discussion points.

## References

- Beaglehole, D., Holzmüller, D., Radhakrishnan, A., Belkin, M. (2026). *xRFM: Accurate, scalable, and interpretable feature learning models for tabular data.* ICLR 2026. arXiv:2508.10053. GitHub: https://github.com/dmbeaglehole/xRFM
- Radhakrishnan, A., Beaglehole, D., Pandit, P., Belkin, M. (2024). *Mechanism for feature learning in neural networks and backpropagation-free machine learning models.* Science 383(6690), 1461–1467.
- Chen, T. & Guestrin, C. (2016). XGBoost.
- Pedregosa, F. et al. (2011). scikit-learn.
- Dua, D. & Graff, C. UCI Machine Learning Repository. https://archive.ics.uci.edu/