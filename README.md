# Habitat Mapping

This repository demonstrates two workflows for land-cover classification using remote sensing data:

- **`habitat_mapping.ipynb`** — End-to-end pipeline including raster preprocessing (clipping, resampling, stacking), label rasterization, and classification with a Random Forest.
- **`deep_learning.ipynb`** — Patch-based semantic segmentation using a U-Net implemented in PyTorch.

---

## Data Workflow

1. **Inputs**
   - **NAIP imagery (2019)** — RGB + NIR
   - **CHM (Canopy Height Model)**
   - **Training polygons** (`classification_polys.shp`)

2. **Preprocessing (`habitat_mapping.ipynb`)**
   - Clip CHM to NAIP extent
   - Resample NAIP to CHM resolution
   - Stack NAIP (4 bands) + CHM (1 band) → 5-band raster
   - Rasterize training polygons to create label raster
   - Save legend mapping (`predicted_labels_rf_legend.csv`)

3. **Classification**
   - Train a **Random Forest** classifier (`sklearn.ensemble.RandomForestClassifier`)
   - Evaluate with precision/recall/F1
   - Save predictions as GeoTIFF (`data/output/predicted_labels_rf_*.tif`)

4. **Visualization**
   - Overlay training polygons on NAIP imagery
   - Generate classified raster plots with legends

---

## Deep Learning Workflow

**`deep_learning.ipynb`** implements a U-Net model:

- Extracts patches from stacked raster + labels
- Applies normalization and optional augmentation
- Uses a **WeightedRandomSampler** for class imbalance
- Trains a U-Net (`in_channels=5`, `n_classes=9`)
- Saves the best model weights (`data/models/habitat_unet.pth`)
- Produces full-scene predictions via sliding-window inference
- Outputs classified raster (`data/output/predicted_classes_dl_9.tif`)

---

## Environment

**Python**: 3.10+ recommended

**Key packages**:
- `rasterio`, `geopandas`, `numpy`, `matplotlib`
- `scikit-learn`
- `torch`, `tqdm`

Install baseline dependencies:
```bash
pip install rasterio geopandas scikit-learn matplotlib tqdm
