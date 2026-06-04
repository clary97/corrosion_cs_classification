# corrosion_cs_classification

> **Original repository:** [beric7/corrosion_cs_classification](https://github.com/beric7/corrosion_cs_classification)
> This repository is a personal working copy based on the original work by Eric Bianchi et al.
> All credit for the original code, dataset, and trained models goes to the original authors — see the [Citation](#citation) section below.

corrosion condition state classification for bridge inspections

<p align="center">
    <img src="/figures/corrosion_pred_with_descriptions.png" | width=800 />
    <img src="/figures/class_color_mapping.png" | width=600 />
</p>

The four semantic classes in the dataset are:
```
Good (Background)
Fair
Poor
Severe
```
***Coming soon in January (?) 2022***
:green_circle:\[[Paper](https://doi.org/10.1061/(ASCE)CP.1943-5487.0001045)] :green_circle:\[[Dataset](https://doi.org/10.7294/16624663.v2)\] :green_circle:\[[Trained models](https://doi.org/10.7294/16628668.v1)\]

The corrosion condition state segmentation dataset which can be used for the localization of structural damage, and for more futuristic style transfer [SPADE](https://arxiv.org/abs/1903.07291) and [GAN](https://arxiv.org/abs/1912.04958) / [GAN-Inversion](https://arxiv.org/abs/2101.05278) applications. 

## Results
We were able to achieve an f1-score of 86.67% using the weighted cross entropy classes model. This included using augmented data. 

<p align="center">
    <img src="/figures/Picture3.jpg"  | width=600/>
</p>

<p align="center">
    <img src="/figures/corr_results.png"  | width=400/>
</p>
    

## Requirements

**Python 3.9 or 3.10** is recommended for best compatibility with all dependencies (particularly `imgaug`).

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

Key packages installed by `requirements.txt`:

| Package | Purpose |
|---|---|
| `numpy`, `scipy`, `scikit-learn` | Numerical / metrics |
| `opencv-python`, `Pillow`, `imageio` | Image I/O |
| `imgaug`, `imgviz` | Data augmentation / visualization |
| `tqdm`, `matplotlib`, `pandas` | Training utilities |
| `visdom` | Live training dashboard |
| `labelme` | Annotation & pre-processing |

## Evaluating the Trained DeeplabV3+ Model
- Clone the respository.
- Download the DeeplabV3+ :green_circle:[trained model weights](https://doi.org/10.7294/16628668.v1)
- Configure ***run_metrics_evaluation.py***

You will get the f1 score, the jaccard index, and the confusion matrix. We suggest running this in an IDE. 
  
## Visualizing the results from the Trained DeeplabV3+ Model
Once training has converged or when it has stopped, we can used the best checkpoint based on the validation data results. This checkpoint is loaded and our test data is evaluated. 

- Clone the respository.

***run_show_results__.py***
- gets predicted masks
- gets combined mask and image overaly
- gets one-hot-encoded vector images of predictions

## Dataset preparation

The training script expects a single directory laid out as follows, where every image has a same-stem mask in the parallel folder:

```
<DATA_DIR>/
  Train/
    Images/  *.jpg | *.jpeg | *.png
    Masks/   *.png   (same stems as Images)
  Test/
    Images/
    Masks/
```

Class color mapping for masks (read by OpenCV in BGR order; equivalent to RGB black / red / green / yellow):

| Index | Class            | BGR value     |
|-------|------------------|---------------|
| 0     | Good (background)| (0, 0, 0)     |
| 1     | Fair             | (0, 0, 128)   |
| 2     | Poor             | (0, 128, 0)   |
| 3     | Severe           | (0, 128, 128) |

Masks must be 512×512 and contain only the four colors above (use `preprocessing/rescale_image.py` and `rescale_segmentation.py` if you need to downscale). If you train on a different class set, edit the `self.mapping` dict and `channels` argument accordingly.

### Rasterizing labelme-polygon datasets (when the source has no pixel masks)

Some datasets ship labelme JSON polygons instead of pre-rendered pixel masks. `preprocessing/rasterize_labelme_dataset.py` converts a labelme dataset into the `Train/Test × Images/Masks` pixel-mask layout used here, baking the four-class BGR colors above into the output and downscaling to 512×512.

1. Open `preprocessing/rasterize_labelme_dataset.py` and edit:
   - `SOURCE_ROOT` — absolute path to the labelme dataset's top level.
   - `SPLIT_SPECS` — list of `(images_subdir, json_subdir, output_split)` tuples. Multiple entries may target the same output split (e.g. merge `validation` into `Train`).
   - `LABEL_TO_BGR` — map each labelme `label` string to a BGR pixel color. Labels not listed here are treated as background.

2. Run:

   ```bash
   python preprocessing/rasterize_labelme_dataset.py \
     --output <RASTERIZED_DIR> \
     --image-size 512
   ```

   The output directory then has the same shape as any pre-rendered dataset and can be fed straight into the unifier below.

### Unifying one or more source datasets

Source datasets rarely come in the exact layout above (folder names differ, file stems collide between datasets, etc.). `preprocessing/prepare_dataset.py` builds the expected layout from one or more sources without modifying the originals — it creates symlinks (or copies) under a destination directory, prefixing filenames so multiple datasets can be merged safely.

1. Open `preprocessing/prepare_dataset.py` and edit the `DATASETS` list. Each entry points at one source dataset's train/test image and mask folders and assigns a short `prefix`. Add another entry per dataset you want to merge.

2. Run the script, passing the unified output directory you want to create:

   ```bash
   python preprocessing/prepare_dataset.py \
     --output <DATA_DIR> \
     --mode symlink     # or `copy` if you need a portable, standalone tree
   ```

3. Pass the same `<DATA_DIR>` to `main_plus.py` via `-data_directory` (see commands below).

## Training with the unified dataset

1. Prepare `<DATA_DIR>` using the steps above.
2. Go into the `training/` folder.
3. If you set up `<DATA_DIR>` correctly you are now ready to begin.

Neccesary and optional inputs to the ***main_plus.py*** file:
('-' means it is neccessary, '--' means that these are optional inputs)
```
 -data_directory = dataset directory path (expects there to be a 'Test' and a 'Train' folder, with folders 'Masks' and 'Images')
 -exp_directory = where the stored metrics and checkpoint weights will be stored
 --epochs = number of epochs
 --batchsize = batch size
 --output_stride = deeplab hyperparameter for output stride
 --channels = number of classes (we have four, the default has been set to four). 
 --class_weights = weights for the cross entropy loss function
 --folder_structure = 'sep' or 'single' (sep = separate (Test, Train), single = only looks at one folder (Train). If you want to get validation results instead of getting back your test dataset results then you should use 'single'. If you want to test directly on the Test dataset then you should use 'sep'.
 --pretrained = if there is a pretrained model to start with then include the path to the model weights here. 
```

Run the following command:
```
python main_plus.py -data_directory '<DATA_DIR>' -exp_directory '<EXP_DIR>' \
--epochs 40 --batch 2
```

During training there are model checkpoints saved every epoch. At these checkpoints the model is compared against the test or validation data. If the test or validation scores are better than the best score, then it is saved.

### Fine-tuning from the published Weighted Cross-Entropy checkpoint

The four pre-trained `.pt` files distributed at the :green_circle:[trained models](https://doi.org/10.7294/16628668.v1) DOI differ only in their loss function (cross-entropy, weighted cross-entropy, L1, L2); all share the DeepLabV3+/ResNet50 architecture, 512×512 input, batch size 2, and horizontal-flip augmentation. The Weighted CE checkpoint (`var_original_wbatch_2_plus_weights_40.pt`, trained with class weights `[0.1, 0.3, 0.3, 0.3]`) is generally the strongest starting point for further fine-tuning.

1. Download the archive and place the checkpoint under any path on disk, e.g.:

   ```
   training/stored_weights/var_original_wwwbatch_2_plus/var_original_wbatch_2_plus_weights_40.pt
   ```

2. From inside `training/`, run:

   ```bash
   python main_plus.py \
     -data_directory '<DATA_DIR>' \
     -exp_directory  '<EXP_DIR>' \
     --epochs 40 --batch 2 \
     --loss cross_entropy \
     --class_weights 0.1 0.3 0.3 0.3 \
     --pretrained 'stored_weights/var_original_wwwbatch_2_plus/var_original_wbatch_2_plus_weights_40.pt'
   ```

   Adjust `--class_weights` to match your dataset's class distribution; the values above are what the original authors used. The `.pt` files themselves are not tracked in this repository — keep them outside git or under a path covered by `.gitignore`. 

## Binary classification experiment (Good vs Corrosion)

As an alternative to 4-class segmentation, Fair/Poor/Severe can be merged into a single **Corrosion** class for a simpler binary task. This often yields higher IoU on the corrosion region since the model no longer needs to distinguish severity boundaries.

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

This remaps Fair/Poor/Severe → BGR (0, 0, 128) while keeping Good as-is, and writes the result to a new `Train/Test` directory tree.

### 2. Train

From inside `training/`:

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
# Per-class metrics + learning curves (compared against 4-class run)
python evaluate_binary.py

# Save 4-panel inference images for the test set
python save_predictions_binary.py
```

---

## Training with a custom dataset

To train on a dataset other than the corrosion condition states (or to extend it):

1. Ensure your image and mask data is 512×512. Use `preprocessing/rescale_image.py` and `rescale_segmentation.py` if you need to downscale. Mask resizing **must** use nearest-neighbor interpolation so new colors aren't introduced.
2. Edit the color map in `training/datahandler_plus.py` (`self.mapping`) so each BGR tuple in your masks maps to the desired class index. Example default for the corrosion dataset:
   ```python
   # 0 = Good (Black), 1 = Fair (Red), 2 = Poor (Green), 3 = Severe (Yellow)
   self.mapping = {(0,0,0): 0, (0,0,128): 1, (0,128,0): 2, (0,128,128): 3}
   ```
3. Pass `--channels N` to `main_plus.py` if your class count differs from 4.
4. Use `preprocessing/prepare_dataset.py` (see [Dataset preparation](#dataset-preparation)) to build the `Train/{Images,Masks}` + `Test/{Images,Masks}` layout that the training script expects.

## Building a Custom Dataset
(The images in the dataset were annotated using [labelme](https://github.com/wkentaro/labelme). We suggest that you use this tool)

0. **If you are planning to extend on the corrosion dataset, then please read the annotation guidelines provided by the author in the :green_circle: [corrosion dataset](https://doi.org/10.7294/16624663.v1) repository.**

1. Before beginning to annotate, we suggest that you use jpeg for the RGB image files. We advised against beginning with images which are already resized. 

2. We have put together a tutorial on tips and tricks on how to use the labelme software in this [youtube video](https://www.youtube.com/watch?v=XtYUPe_JfRw). We also made a [video on youtube](https://www.youtube.com/watch?v=Zd4YmSMLYFQ) showing how to set up labelme with Anaconda prompt.

3. After annotating you will have matching JSON and jpeg files, indicating the annotation and image pair respectfully. 

4. You will take these files and generate masks and one-hot-encoded vector files using ***run_labelme2voc_.py*** file in preprocessing. Then you can re-scale these images and masks using the respective files in preprocessing. You can also use the random sort function we have created to randomly split the data. 

The ***labels_corrosion_segmentation.txt*** file contains the class labels needed for the ***run_labelme2voc_.py*** function. If your classes are different then they need to be reflected in this particular file.

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
author = {Eric Bianchi  and Matthew Hebdon },
title = {Development of Extendable Open-Source Structural Inspection Datasets},
journal = {Journal of Computing in Civil Engineering},
volume = {36},
number = {6},
pages = {04022039},
year = {2022},
doi = {10.1061/(ASCE)CP.1943-5487.0001045},
```


