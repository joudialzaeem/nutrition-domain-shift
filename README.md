# Nutritional Estimation Under Domain Shift

Code and experimental results for my MSc Data Science Research Module at the University of Potsdam.

This project investigates how well image-based nutritional estimation models generalize from Nutrition5k to FoodBD. I trained three pretrained architectures—ResNet50, EfficientNet-B0 and ViT-B/16—to predict calories, protein, carbohydrates and fat from RGB food images. I then evaluated them on FoodBD and investigated whether unsupervised domain adaptation using Maximum Mean Discrepancy (MMD) could improve their performance.

## Datasets

### Nutrition5k

- [Official Nutrition5k repository](https://github.com/google-research-datasets/Nutrition5k)
- [Google Cloud Storage download](https://console.cloud.google.com/storage/browser/nutrition5k_dataset)

Nutrition5k was used as the labelled source dataset. I used overhead RGB images and the corresponding nutritional metadata, rather than depth images.

The original overhead split was used to select the images. After excluding unavailable and problematic entries, the curated dataset contained 3,227 dishes:

| Split | Images |
|---|---:|
| Training | 2,245 |
| Validation | 484 |
| Test | 498 |

The official training/test separation was preserved. The training pool was divided into training and validation sets using a collection-date-based grouped split. Nutritional targets were standardized using statistics calculated from the training set only.

### FoodBD

- [FoodBD dataset — Mendeley Data, Version 2](https://data.mendeley.com/datasets/xh3ghf3jbg/2)

FoodBD was used to evaluate generalization to a different dataset of smartphone food photographs. The original dataset contains 1,837 images with nutritional annotations. Four entries could not be matched to the dataset metadata, leaving 1,833 images for this project.

| Split | Images |
|---|---:|
| Training | 1,288 |
| Validation | 365 |
| Test | 180 |

The source-trained models were initially evaluated on all 1,833 prepared FoodBD images. For the reported external test results and before/after MMD comparisons, I used only the 180 held-out test images.

The 1,653 FoodBD training and validation images were used without nutritional labels during MMD adaptation.

Both original datasets are available under Creative Commons Attribution 4.0 (CC BY 4.0). Please refer to their original repositories and publications for attribution and dataset details.

## Models and experimental pipeline

Three ImageNet-pretrained architectures were used:

- ResNet50
- EfficientNet-B0
- ViT-B/16

Each model was trained for four-output nutritional regression on the curated Nutrition5k dataset.

The main experiments consisted of source-domain training, direct FoodBD evaluation, MMD adaptation and evaluation of the adapted models on the same 180 FoodBD test images.

The scripts are organized as follows:

| Script | Purpose |
|---|---|
| `prepare_nutrition5k_data.py` | Apply the final quality exclusions and generate splits and the target scaler from the selected source metadata |
| `prepare_foodbd_data.py` | Prepare the 1,833-image FoodBD metadata table |
| `train_model.py` | Train the three source-domain models |
| `evaluate_foodbd.py` | Evaluate the source-trained models on FoodBD |
| `adapt_mmd.py` | Perform unsupervised MMD adaptation |
| `evaluate_mmd_foodbd.py` | Evaluate adapted models on the held-out FoodBD test set |
| `analyze_mmd_statistics.py` | Calculate paired errors, bootstrap intervals, Wilcoxon tests and Holm correction |
| `analyze_nutritional_shift.py` | Analyze nutritional distance and prediction errors |
| `run_foodbd_image_diagnostics_verified.py` | Analyze image characteristics and their associations with prediction errors |
| `make_foodbd_diagnostic_figures.py` | Generate selected diagnostic plots and example images |
| `evaluate_five_meals.py` | Run the six trained model variants on 25 additional meal photographs |
| `analyze_five_meals.py` | Summarize five-meal errors and photography-condition effects |

The detailed sequence of commands is documented in `RUN_ORDER.md`.

## Main results

All three models performed substantially worse on FoodBD than on the Nutrition5k test set, showing a considerable generalization gap.

MMD adaptation improved some outputs, but the improvements were not consistent across architectures or nutritional targets.

After Holm correction for the 12 model-target comparisons, three changes were statistically significant:

| Model | Nutrient | Before MAE | After MAE | Result |
|---|---|---:|---:|---|
| EfficientNet-B0 | Protein | 18.69 g | 17.21 g | Improved |
| ViT-B/16 | Protein | 16.01 g | 12.42 g | Improved |
| ViT-B/16 | Carbohydrates | 17.17 g | 20.55 g | Worsened |

All other comparisons were not statistically significant after Holm correction.

Additional analyses examined nutritional differences between datasets and image characteristics such as brightness, contrast, dark-pixel fraction and edge density. The five-meal experiment was used as a small descriptive robustness study and was not intended to provide statistically generalizable conclusions.

## Repository contents

- `code/` — 12 Python scripts
- `slurm/` — six original HPC submission scripts
- `data/` — selected metadata, prepared splits and target standardization
- `results/` — saved predictions, evaluation metrics and statistical results
- `figures/` — selected diagnostic figures
- `RUN_ORDER.md` — execution sequence
- `requirements.txt` — Python dependencies

The repository contains the per-image source-model predictions for all 1,833 FoodBD images and MMD-model predictions for the held-out 180-image test set.

The original paired statistical results are saved in `results/mmd_before_after_statistics.csv`.

## Reproducing the analyses

After installing the dependencies listed in `requirements.txt`, the saved MMD prediction results can be analyzed without retraining the models:

```bash
python code/analyze_mmd_statistics.py
```

The nutritional-shift analysis can be reproduced from the included prepared data:

```bash
python code/analyze_nutritional_shift.py
```

The original image diagnostics require the corresponding FoodBD test photographs. Reproducing model training and inference requires the original dataset images and a compatible PyTorch environment.

Expected locations for these additional inputs are:

- Nutrition5k images: `data/images/`
- FoodBD images: `data/foodbd_eval_1833/images/`
- Trained model checkpoints: `checkpoints/`
- FoodBD test-image archive: `data/foodbd_test_images.zip`
- Additional five-meal images: `five_meal_study/images/5 meals images/`

The code uses the image paths recorded in the prepared metadata, so the original directory structure and file names need to be preserved.

The six trained `.pt` checkpoints, original image archives and full HPC execution logs are not included in this compact repository because of their size. They have been retained separately in the original project backup.

The Nutrition5k preparation script starts from the selected metadata used for this project; it does not independently download and select all images from the complete original dataset.

## Computing environment

The original experiments were run on the University of Potsdam HPC cluster using:

- NVIDIA Tesla V100 GPU (16 GB)
- PyTorch 2.5.1+cu121
- torchvision 0.20.1+cu121
- Conda environment: `nutritionml`

The main dependencies include PyTorch, torchvision, NumPy, pandas, Pillow, scikit-learn, SciPy, statsmodels, OpenCV and Matplotlib.

The original SLURM scripts are included to document the HPC jobs. They contain cluster-specific paths and environment settings and may need to be adjusted when running on another system.

## Dataset attribution

**Nutrition5k:** Thames, Q., Karpur, A., Norris, W., Xia, F., Panait, L., Weyand, T., & Sim, J. (2021). *Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food.* Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8903–8911.

**FoodBD:** Ahmed, B., Haque, M. E., Huq, A. K. O., Ali, M. E., Masud, M., & Naznin, M. *FoodBD: A Polygon-Annotated Meal Image Dataset of Bangladeshi Cuisine with Visual and Nutritional Labels.* Mendeley Data, Version 2. https://doi.org/10.17632/xh3ghf3jbg.2

Both datasets are credited to their original authors and remain subject to their respective CC BY 4.0 terms.
