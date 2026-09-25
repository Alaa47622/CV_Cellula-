# Unified Eye Disease Classification

8-class fundus image classification (Normal, Diabetic Retinopathy, Others, Glaucoma,
Cataract, Myopia, AMD, Hypertension) on a Kaggle-extracted eye disease dataset. The
project builds a CNN from scratch, fine-tunes a pretrained ResNet50, and compares
the two on accuracy, per-class F1, model size, and inference speed.

## Notebooks

Run in order:

| Notebook | What it does |
|---|---|
| `00_setup_generate_splits.ipynb` | Builds `train.csv` / `test.csv` from `metadata.csv`, matching the per-class counts in `catalog_summary.csv` (`val.csv` is assumed to already exist). Run once before anything else. |
| `01_data_exploration.ipynb` | Loads the splits, summarizes class counts, plots class distribution and imbalance, visualizes sample images per class, and computes class weights. |
| `02_scratch_cnn.ipynb` | Trains a CNN from scratch (no pretrained weights) with class-weighted loss. Saves model, training curves, classification report, confusion matrix, and inference timing to `artifacts/`. |
| `03_transfer_learning.ipynb` | Fine-tunes an ImageNet-pretrained ResNet50 in two stages (frozen-backbone warm-up, then partial unfreeze at low LR). Saves the same set of artifacts as `02`. |
| `04_model_comparison.ipynb` | Loads the saved metrics from `02` and `03` and compares accuracy, macro/weighted F1, per-class F1 vs. support, model size, and throughput. |

## Data layout

The notebooks expect (adjust the `CONFIG` dict at the top of `01_data_exploration.ipynb`
and the `DATA_DIR` in `00_setup_generate_splits.ipynb` to match your paths):

```
data/
 ├── metadata.csv
 ├── catalog_summary.csv
 └── splits/
      ├── train.csv
      ├── val.csv
      └── test.csv
images/
 └── <image files referenced by the CSVs>
```

Paths currently default to a Kaggle working directory (`/kaggle/working/dataset_extracted/...`);
update `CONFIG["DATA_DIR"]` / `CONFIG["IMG_DIR"]` in each notebook if running elsewhere.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then launch Jupyter and run the notebooks in the order listed above:

```bash
jupyter notebook
```

## Outputs

Each training notebook writes its model, plots, and metrics JSON to `artifacts/`
(git-ignored by default). `04_model_comparison.ipynb` reads those JSON files, so
`02` and `03` must be run first.

## Class imbalance

The dataset is heavily imbalanced (majority class *Normal* ≈ 4,698 images vs.
minority class *Hypertension* ≈ 88 images, ~53:1). Both training notebooks use
inverse-frequency class weighting to address this; `01_data_exploration.ipynb`
and `04_model_comparison.ipynb` both examine its effect on per-class metrics.
