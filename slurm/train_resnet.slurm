#!/bin/bash
#SBATCH --job-name=resnet50
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/work/alzaeem1/nutrition_project/logs/resnet50_%j.out

cd /work/alzaeem1/nutrition_project

/home/alzaeem1/.conda/envs/nutritionml/bin/python \
    code/train_model.py \
    --model resnet50
