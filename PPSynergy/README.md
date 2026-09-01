# PPSynergy

PPSynergy is a drug synergy prediction framework based on pathway-level functional
perturbation representations. It uniformly maps the functional changes induced by two
drugs and the cell-line background into 2D images at the pathway level. A
diffusion-enhanced module reconstructs and reinforces the pathway perturbation patterns
related to drug action, and a convolutional neural network then models the interactions
between the two drug perturbations and the cellular environment to predict the synergy
of drug combinations. Compared with methods that rely only on molecular structures or
gene expression, PPSynergy focuses on how drugs jointly alter functional pathways in a
specific cellular context, and its prediction rationale can be explained through
pathway attribution analysis.

For each sample `(drug1, drug2, cell line)`, three independent matrices are built and
stacked as the channels of one image:

- **Channel 0** — drug 1 pathway activity matrix
- **Channel 1** — drug 2 pathway activity matrix
- **Channel 2** — cell line pathway expression matrix

Each channel is Z-score normalized before stacking, producing a tensor of shape
`(num_samples, 3, 384, 196)`. A 5-fold cross-validation pipeline trains the model and
reports per-fold and mean ± std metrics.

---

## Requirements

- **Python** = 3.11
- **CUDA** 12.8 (GPU recommended) or CPU
- **NVIDIA GPU** NVIDIA GeForce RTX 5080

### Core Dependencies

| Package        | Version   | Purpose                                       |
|----------------|-----------|-----------------------------------------------|
| PyTorch        | 2.7.1+cu128 | Deep learning framework                     |
| NumPy          | 1.26.4    | Numerical computing                           |
| Pandas         | 2.3.1     | Data I/O and manipulation                     |
| scikit-learn   | 1.7.1     | Evaluation metrics, data splitting            |
| matplotlib     | 3.9.2     | Result visualization                          |

---

## Project Structure

```
PPSynergy/
├── train.py                    # Training, 5-fold CV and result summary
├── PPSynergy.py                # PPSynergyNet
├── model_function.py           # train / test / predict / metric functions
├── README.md
├── data/
│   ├── creat.py                # 3-channel image generation script
│   ├── orthogonality.csv       # Pathway coordinates (label, x, y)
│   ├── ONEIL/                  # ONEIL dataset files
│   ├── NCI-ALMANAC/            # NCI-ALMANAC dataset files
│   └── Construction of cell line features/
│       ├── cell_pathway_mean_scores.py   # Cell-line pathway activity scoring
│       └── MSigDB.gmt                   # Pathway-gene mapping (GMT)
└── result/
    ├── ONEIL/                  # Training outputs for ONEIL
    └── NCI-ALMANAC/            # Training outputs for NCI-ALMANAC
```

---

## Getting Started

### 1. Install dependencies

```bash
conda create -n ppsynergy python=3.11 -y
conda activate ppsynergy

# Install PyTorch with CUDA 12.8 (or CPU-only build if no GPU)
pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128

pip install numpy==1.26.4 pandas==2.3.1 scikit-learn==1.7.1 matplotlib==3.9.2
```

### 2. (Optional) Build cell-line pathway features

If you need to generate new cell-line pathway activity scores from a gene
expression matrix and the MSigDB GMT file:

```bash
python "data/Construction of cell line features/cell_pathway_mean_scores.py"
```

Inputs (edit the paths at the top of the script to match your files):
- Gene expression matrix: `gene symbol x cell lines (TPM)` (CSV)
- Pathway list: `unique_pathways.csv` (any number of pathways)
- Pathway-gene mapping: `MSigDB.gmt` (tab-separated; fixed file next to the script)

Output:
- `cellline11_pathway_scores.csv` — columns `pathway + cell lines`, rounded to 4 decimals

### 3. Build the image data

Generate the 3-channel images with `data/creat.py`. Choose the dataset with
`--dataset`:

```bash
# ONEIL dataset -> data/ONEIL.npy
python data/creat.py --dataset ONEIL

# NCI-ALMANAC dataset -> data/NCI-ALMANAC.npy
python data/creat.py --dataset NCI-ALMANAC
```

What `creat.py` reads (all under `data/`):

| Purpose                  | ONEIL                       | NCI-ALMANAC                     |
|--------------------------|-----------------------------|---------------------------------|
| Pathway coordinates      | `orthogonality.csv`         | `orthogonality.csv`             |
| Drug-pathway mapping     | `ONEIL/drug_pathway.csv`    | `NCI-ALMANAC/drugs_pathways.csv`|
| Cell-line expression     | `ONEIL/cell-line.csv`       | `NCI-ALMANAC/cell_line.csv`     |
| Sample data (drug1,drug2,cell,label) | `ONEIL/ONEIL_socre.csv` | `NCI-ALMANAC/NCI-ALMANAC_socre.csv` |
| Pathway list             | `ONEIL/pathways.csv`        | `NCI-ALMANAC/pathways.csv`      |
| Pathway activity (optional) | `ONEIL/State_predict.csv` | `NCI-ALMANAC/State_predict.csv`|

> **Note**: `State_predict.csv` is optional. If it is missing, the script prints a warning and falls back to **binary counting**
> (no activity mapping) for the drug matrices.

Output: `data/{DATASET}.npy` with shape `(num_samples, 3, 384, 196)`.

### 4. Train

Run the training script with the target dataset:

```bash
# ONEIL (default; automatically uses AUC+ACC for best-model selection)
python train.py --dataset ONEIL

# NCI-ALMANAC (automatically uses AUPR+F1 for best-model selection)
python train.py --dataset NCI-ALMANAC

# Explicitly choose the best-model selection metric
python train.py --dataset ONEIL --metric AUC+ACC
python train.py --dataset NCI-ALMANAC --metric AUPR+F1
```

The best-model selection metric is auto-selected by dataset unless `--metric`
is given:

| Dataset       | Default selection metric |
|---------------|--------------------------|
| ONEIL         | `AUC + ACC`              |
| NCI-ALMANAC   | `PR_AUC + F1`            |

The pipeline performs **5-fold cross-validation**; inside each fold the model with
the highest selection score on the test split is saved as the best model.

### 5. Outputs

All outputs are written to `result/{DATASET}/`:

| File                          | Description                                    |
|-------------------------------|------------------------------------------------|
| `_{fold}--AUCs--.txt`         | Per-epoch metrics (AUC, PR_AUC, ACC, BACC, ...)|
| `_{fold}.txt`                 | Per-epoch train/test loss & accuracy           |
| `_{fold}_best.pt`             | Best model checkpoint of each fold             |
| `summary_best_metrics.txt`    | Best metrics per fold (tabular)                |

In addition, the console prints the best metrics of every fold and the
**mean ± std over the 5 folds** (4 decimal places) for all metrics including the
selection metric.

---
<!-- 
## Authors

This work is developed and maintained by:

- **Yiwei Chen** — Anhui University  -->
