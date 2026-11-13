## Data preparation

With the required source metadata and image datasets available:

```bash
python code/prepare_nutrition5k_data.py
python code/prepare_foodbd_data.py
```

# Run order

From the project root:

```bash
python code/train_model.py --model resnet50
python code/train_model.py --model efficientnet_b0
python code/train_model.py --model vit_b_16

python code/evaluate_foodbd.py --model resnet50
python code/evaluate_foodbd.py --model efficientnet_b0
python code/evaluate_foodbd.py --model vit_b_16

python code/adapt_mmd.py --model resnet50
python code/adapt_mmd.py --model efficientnet_b0
python code/adapt_mmd.py --model vit_b_16

python code/evaluate_mmd_foodbd.py --model resnet50
python code/evaluate_mmd_foodbd.py --model efficientnet_b0
python code/evaluate_mmd_foodbd.py --model vit_b_16

python code/analyze_mmd_statistics.py
python code/analyze_nutritional_shift.py
python code/run_foodbd_image_diagnostics_verified.py
python code/make_foodbd_diagnostic_figures.py
python code/analyze_five_meals.py
```

`evaluate_five_meals.py` requires the 25 personal meal photographs and the six frozen checkpoints in the original project layout.

`run_foodbd_image_diagnostics_verified.py` requires the FoodBD test-image ZIP described in the README.


To regenerate the five-meal predictions after placing the
25 photographs and six checkpoints in their expected folders:

```bash
python code/evaluate_five_meals.py
python code/analyze_five_meals.py
```
