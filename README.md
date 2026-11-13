# Nutritional Estimation Under Domain Shift — Code Submission

This package contains the cleaned code and compact reproducibility material for the MSc Data Science research module evaluating nutritional estimation under domain shift from Nutrition5k to FoodBD.

## Main experimental pipeline

1. `prepare_nutrition5k_data.py`
   Reproduces the final Nutrition5k curation, grouped train/validation split, and training-only target scaler.

2. `train_model.py`
   Trains ImageNet-pretrained ResNet50, EfficientNet-B0, and ViT-B/16 for four-output regression: calories, protein, carbohydrates, and fat.

3. `prepare_foodbd_data.py`
   Reproduces the final 1,833-image nutrition-labelled FoodBD metadata table.

4. `evaluate_foodbd.py`
   Evaluates the frozen source-trained models on FoodBD.

5. `adapt_mmd.py`
   Performs unsupervised MMD feature alignment using labelled Nutrition5k source data and FoodBD train+valid images without nutrition labels.

6. `evaluate_mmd_foodbd.py`
   Evaluates the adapted models on the untouched 180-image FoodBD test set.

7. `analyze_mmd_statistics.py`
   Reproduces the paired bootstrap, Wilcoxon signed-rank tests, and Holm correction using the baseline and MMD prediction CSVs included in the results folder.

8. `analyze_nutritional_shift.py`
   Reproduces the final target-distribution diagnostic.

9. `run_foodbd_image_diagnostics_verified.py`
   Recomputes the verified FoodBD image-quality proxies and 24 Spearman tests with Holm correction.

10. `make_foodbd_diagnostic_figures.py`
    Recreates the two brightness scatterplots and the selected-case contact sheet.

11. `evaluate_five_meals.py`
    Runs the six frozen model variants on the 25 personal meal photographs.

12. `analyze_five_meals.py`
    Reproduces the descriptive five-meal error and photography-condition summaries.

## Frozen dataset choices

Nutrition5k:
- 3,227 final cleaned dishes
- 2,245 train
- 484 validation
- 498 untouched test
- target standardization fitted on the training split only

FoodBD:
- 1,833 nutrition-labelled images
- 1,288 train
- 365 validation
- 180 held-out test
- train+valid images were used without nutrition labels for MMD
- the 180-image test set was reserved for final evaluation

## Verified diagnostic results

The nutritional-shift script reproduces:
- overall nutritional distance vs aggregate standardized prediction error: Spearman rho = 0.46244
- carbohydrate distance vs carbohydrate error: rho = 0.42844
- closest quartile: 132.67 kcal and 11.78 g carbohydrate mean absolute error
- furthest quartile: 199.42 kcal and 38.00 g carbohydrate mean absolute error

The verified image diagnostics are included under `results/image_diagnostics/`.

The five-meal analysis is descriptive only. It uses five meals, five photo conditions, and six frozen model variants.

## MMD statistical validation

The repository includes the original FoodBD prediction CSVs for
all three source-trained models (1,833 images each), along with
the MMD-adapted predictions for the 180 held-out test images.

The paired results are in
`results/mmd_before_after_statistics.csv`.

`code/analyze_mmd_statistics.py` can reproduce the paired
bootstrap confidence intervals, Wilcoxon tests, and Holm
correction using these saved prediction files.

After Holm correction, the significant findings are:
- EfficientNet-B0: protein MAE improved
- ViT-B/16: protein MAE improved
- ViT-B/16: carbohydrate MAE worsened

No other model-target comparison was significant after
Holm correction.

## Included files and reproducibility

The repository includes:
- 12 Python scripts in `code/`
- Six original HPC SLURM scripts in `slurm/`
- Prepared dataset metadata and split assignments in `data/`
- Saved model predictions, statistical results, and diagnostics
  in `results/`
- Selected figures in `figures/`
- `requirements.txt`, `RUN_ORDER.md`, and this README

The original SLURM scripts retain the paths and Conda environment
used on the University of Potsdam HPC. They document the original
jobs and may need changes before running on another system.

The full image datasets, six trained `.pt` model checkpoints,
and original HPC execution logs are not included in this compact
repository. They are retained separately in the original project
backup.

Expected locations for additional inputs:
- Nutrition5k images: `data/images/`
- FoodBD images: `data/foodbd_eval_1833/images/`
- Model checkpoints: `checkpoints/`
- FoodBD diagnostic image ZIP: `data/foodbd_test_images.zip`
- Five-meal photos:
  `five_meal_study/images/5 meals images/`

The prepared metadata and saved predictions allow the reported
statistics to be checked without retraining the models.
The full training and inference pipeline requires the omitted
images, appropriate model checkpoints, and a compatible
PyTorch environment.

The Nutrition5k preparation script starts from the selected
metadata supplied in this repository. It does not independently
recreate the original dish selection from the raw dataset download.

Dataset and metadata redistribution conditions should be checked
before making the repository public.

## HPC environment

University of Potsdam HPC
Tesla V100, 16 GB
PyTorch 2.5.1+cu121
torchvision 0.20.1+cu121
Conda environment: `nutritionml`

Main Python dependencies:
PyTorch, torchvision, pandas, NumPy, Pillow, scikit-learn, SciPy, statsmodels, OpenCV, Matplotlib.


## GitHub submission layout

This version includes the original six SLURM scripts in `slurm/`.
They retain the HPC-specific paths used for the original runs.

The prepared Nutrition5k split is located at:
`data/nutrition5k_model_splits.csv`

The prepared FoodBD table is located at:
`data/foodbd_eval_1833/foodbd_evaluation_1833.csv`

Full original-model FoodBD predictions, MMD predictions and the
paired statistical results are included in `results/`.

Raw dataset photographs and the six trained `.pt` checkpoints
are not included in this compact package. Re-running inference
or training requires those files to be placed in the directories
expected by the scripts.

The five-meal analysis outputs are in `results/five_meal/`.
The original photos are required only to rerun inference.

The original HPC backup has been retained separately.
