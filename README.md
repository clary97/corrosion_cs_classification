# corrosion_cs_classification

> **Original repository:** [beric7/corrosion_cs_classification](https://github.com/beric7/corrosion_cs_classification)  
> This is a personal fork of the original work by Eric Bianchi et al. (Virginia Tech).  
> The original code, dataset, and trained models belong to the original authors — see the [Citation](#citation) section.  
> This fork adds dataset unification utilities, a binary classification experiment, and various bug fixes.

Semantic segmentation of corrosion condition states for bridge inspection, using DeepLabV3+ with a ResNet-101 backbone.

<p align="center">
    <img src="/figures/corrosion_pred_with_descriptions.png" | width=800 />
    <img src="/figures/class_color_mapping.png" | width=600 />
</p>

The four semantic classes are:

| Class | Description |
|---|---|
| Good | No visible corrosion (background) |
| Fair | Light surface corrosion |
| Poor | Moderate corrosion |
| Severe | Heavy corrosion with section loss |

:green_circle: [Paper](https://doi.org/10.1061/(ASCE)CP.1943-5487.0001045) &nbsp;|&nbsp; :green_circle: [Dataset](https://doi.org/10.7294/16624663.v2) &nbsp;|&nbsp; :green_circle: [Trained models](https://doi.org/10.7294/16628668.v1)

---

## Original results (Bianchi et al.)

The original authors report an F1-score of **86.67%** using weighted cross-entropy with augmented data.

<p align="center">
    <img src="/figures/Picture3.jpg" | width=600/>
</p>
<p align="center">
    <img src="/figures/corr_results.png" | width=400/>
</p>

---

## Fork results

### 4-class segmentation (Good / Fair / Poor / Severe)

Trained for 40 epochs on the unified dataset (CCSD + CIR, ~900 train / 56 test images).

| Class | Precision | Recall | F1 | IoU | FPR |
|---|---|---|---|---|---|
| Good | 0.938 | 0.959 | 0.949 | 0.902 | 0.204 |
| Fair | 0.625 | 0.563 | 0.592 | 0.421 | 0.045 |
| Poor | 0.664 | 0.691 | 0.678 | 0.512 | 0.035 |
| Severe | 0.832 | 0.561 | 0.670 | 0.504 | 0.003 |
| **Weighted avg** | **0.874** | **0.877** | **0.874** | **0.799** | **0.164** |

<p align="center">
    <img src="/figures/4class_training_curves.png" width=800/>
</p>
<p align="center">
    <img src="/figures/4class_evaluation_metrics.png" width=800/>
</p>

**Prediction samples** (Input | Ground Truth | Prediction | Overlay):

<p align="center">
    <img src="/figures/4class_pred_sample1.png" width=800/>
</p>
<p align="center">
    <img src="/figures/4class_pred_sample2.png" width=800/>
</p>

---

### Binary segmentation (Good vs Corrosion)

Fair / Poor / Severe merged into a single **Corrosion** class. Trained for 40 epochs with the same architecture.

| Class | Precision | Recall | F1 | IoU | FPR |
|---|---|---|---|---|---|
| Good | 0.957 | 0.949 | 0.953 | 0.910 | 0.139 |
| Corrosion | 0.840 | 0.861 | 0.850 | 0.740 | 0.051 |
| **Weighted avg** | **0.929** | **0.928** | **0.929** | **0.870** | **0.118** |

Compared to the 4-class model: F1 **+5.44%p**, IoU **+7.06%p**, FPR **−4.56%p**.

<p align="center">
    <img src="/figures/binary_training_curves.png" width=800/>
</p>
<p align="center">
    <img src="/figures/binary_evaluation_metrics.png" width=800/>
</p>

**Prediction samples** (Input | Ground Truth | Prediction | Overlay):

<p align="center">
    <img src="/figures/binary_pred_sample1.png" width=800/>
</p>
<p align="center">
    <img src="/figures/binary_pred_sample2.png" width=800/>
</p>
<p align="center">
    <img src="/figures/binary_pred_sample3.png" width=800/>
</p>

---

## Requirements

**Python 3.9 or 3.10** is recommended for best compatibility with all dependencies.

### 1. Create and activate a virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows
```

### 2. Install PyTorch with GPU (CUDA) support

Choose the command that matches your CUDA version (check with `nvidia-smi`):

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `numpy`, `scipy`, `scikit-learn` | Numerical / metrics |
| `opencv-python`, `Pillow`, `imageio` | Image I/O |
| `albumentations` | Data augmentation |
| `tqdm`, `matplotlib`, `pandas` | Training utilities |
| `visdom` | Live training dashboard |
| `labelme` | Annotation & pre-processing |

---

## Dataset preparation

The training script expects a single directory in the following layout:

```
<DATA_DIR>/
  Train/
    Images/  *.jpg | *.jpeg | *.png
    Masks/   *.png   (same stems as Images)
  Test/
    Images/
    Masks/
```

Class color mapping for masks (OpenCV reads in BGR order):

| Index | Class | BGR value |
|---|---|---|
| 0 | Good (background) | (0, 0, 0) |
| 1 | Fair | (0, 0, 128) |
| 2 | Poor | (0, 128, 0) |
| 3 | Severe | (0, 128, 128) |

Masks must be 512×512. Use `preprocessing/rescale_image.py` and `preprocessing/rescale_segmentation.py` to downscale if needed. Mask resizing **must** use nearest-neighbor interpolation to avoid introducing new colors.

### Rasterizing labelme-polygon datasets

Some datasets ship labelme JSON polygons instead of pixel masks. `preprocessing/rasterize_labelme_dataset.py` converts a labelme dataset into the `Train/Test × Images/Masks` layout.

1. Edit `SOURCE_ROOT`, `SPLIT_SPECS`, and `LABEL_TO_BGR` inside the script.

2. Run:

   ```bash
   python preprocessing/rasterize_labelme_dataset.py \
     --output <RASTERIZED_DIR> \
     --image-size 512
   ```

### Unifying multiple source datasets

`preprocessing/prepare_dataset.py` merges one or more datasets into the expected layout without modifying the originals.

1. Edit the `DATASETS` list in the script (root path, split folders, and prefix per dataset).

2. Run:

   ```bash
   python preprocessing/prepare_dataset.py \
     --output <DATA_DIR> \
     --mode symlink     # or `copy` for a portable standalone tree
   ```

3. Pass `<DATA_DIR>` to `main_plus.py` via `-data_directory`.

---

## Training (4-class)

From inside `training/`:

```bash
python main_plus.py \
  -data_directory '<DATA_DIR>' \
  -exp_directory  '<EXP_DIR>' \
  --epochs 40 --batchsize 2 --output_stride 8 --channels 4 \
  --loss cross_entropy \
  --class_weights 0.1 0.3 0.3 0.3
```

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `-data_directory` | ✅ | Path to dataset root (must contain `Train/` and `Test/`) |
| `-exp_directory` | ✅ | Path to save checkpoints and logs |
| `--epochs` | | Number of training epochs (default: 10) |
| `--batchsize` | | Batch size (default: 2) |
| `--output_stride` | | DeepLabV3+ output stride (default: 8) |
| `--channels` | | Number of output classes (default: 4) |
| `--class_weights` | | Per-class weights for cross-entropy loss |
| `--pretrained` | | Path to a `.pt` checkpoint to fine-tune from |

Checkpoints are saved when the test score improves.

### Fine-tuning from the published checkpoint

The original authors provide four pre-trained checkpoints at the :green_circle:[trained models](https://doi.org/10.7294/16628668.v1) DOI (cross-entropy, weighted cross-entropy, L1, L2). The weighted cross-entropy checkpoint is generally the strongest starting point.

1. Download and place the checkpoint, e.g.:
   ```
   training/stored_weights/var_original_wwwbatch_2_plus/var_original_wbatch_2_plus_weights_40.pt
   ```

2. From inside `training/`, run:

   ```bash
   python main_plus.py \
     -data_directory '<DATA_DIR>' \
     -exp_directory  '<EXP_DIR>' \
     --epochs 40 --batchsize 2 --output_stride 8 --channels 4 \
     --loss cross_entropy \
     --class_weights 0.1 0.3 0.3 0.3 \
     --pretrained 'stored_weights/var_original_wwwbatch_2_plus/var_original_wbatch_2_plus_weights_40.pt'
   ```

### Evaluate and save predictions

```bash
# Learning curves + per-class Precision / Recall / F1 / IoU / FPR
python evaluate_results.py

# Save 4-panel inference images for the test set
python save_predictions.py
```

---

## Binary classification experiment (Good vs Corrosion)

Fair/Poor/Severe can be merged into a single **Corrosion** class for a binary segmentation task. This simplifies the problem and typically yields higher IoU on the corrosion region.

| Class | Index | Mask color (BGR) |
|---|---|---|
| Good | 0 | (0, 0, 0) |
| Corrosion | 1 | (0, 0, 128) |

### 1. Prepare the binary dataset

```bash
python preprocessing/prepare_binary_dataset.py \
  --input  /path/to/unified_corrosion \
  --output /path/to/unified_corrosion_binary
```

### 2. Train

```bash
python main_plus.py \
  -data_directory '/path/to/unified_corrosion_binary' \
  -exp_directory  './stored_weights/finetune_binary_v1' \
  --epochs 40 --batchsize 2 --output_stride 8 --channels 2 \
  --loss cross_entropy \
  --class_weights 0.15 0.85
```

### 3. Evaluate and save predictions

```bash
# Per-class metrics + comparison curves against 4-class run
python evaluate_binary.py

# Save 4-panel inference images
python save_predictions_binary.py
```

---

## Training with a custom dataset

1. Ensure images and masks are 512×512 (use scripts in `preprocessing/` if needed).
2. Edit `self.mapping` in `training/datahandler_plus.py` to match your BGR color scheme:
   ```python
   # Example: 4-class corrosion mapping
   self.mapping = {(0,0,0): 0, (0,0,128): 1, (0,128,0): 2, (0,128,128): 3}
   ```
3. Pass `--channels N` if your class count differs from 4.
4. Use `preprocessing/prepare_dataset.py` to build the expected directory layout.

---

## Annotation

The original dataset was annotated with [labelme](https://github.com/wkentaro/labelme). Annotation guidelines are provided by the original authors in the :green_circle: [corrosion dataset](https://doi.org/10.7294/16624663.v1) repository.

To generate pixel masks from labelme JSON files, use `preprocessing/run_labelme2voc_.py` with `preprocessing/labels_corrosion_segmentation.txt` as the class label file.

---

## Citation

Corrosion Condition State Dataset:
```
Bianchi, Eric; Hebdon, Matthew (2021): Corrosion Condition State Semantic Segmentation Dataset.
University Libraries, Virginia Tech. Dataset. https://doi.org/10.7294/16624663.v2
```

Corrosion Condition State Model:
```
Bianchi, Eric; Hebdon, Matthew (2021): Trained Model for the Semantic Segmentation of Structural Material.
University Libraries, Virginia Tech. Software. https://doi.org/10.7294/16628620.v1
```

Paper:
```
@article{doi:10.1061/(ASCE)CP.1943-5487.0001045,
  author  = {Eric Bianchi and Matthew Hebdon},
  title   = {Development of Extendable Open-Source Structural Inspection Datasets},
  journal = {Journal of Computing in Civil Engineering},
  volume  = {36},
  number  = {6},
  pages   = {04022039},
  year    = {2022},
  doi     = {10.1061/(ASCE)CP.1943-5487.0001045},
}
```
