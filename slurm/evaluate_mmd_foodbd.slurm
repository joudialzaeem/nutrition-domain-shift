#!/bin/bash
#SBATCH --job-name=mmd_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/work/alzaeem1/nutrition_project/logs/mmd_eval_%j.out
#SBATCH --error=/work/alzaeem1/nutrition_project/logs/mmd_eval_%j.err

set -e

cd /work/alzaeem1/nutrition_project

/home/alzaeem1/.conda/envs/nutritionml/bin/python \
    code/evaluate_mmd_foodbd.py \
    --model "$1"
