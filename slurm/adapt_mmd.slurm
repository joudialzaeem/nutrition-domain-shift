#!/bin/bash
#SBATCH --job-name=mmd_adapt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/work/alzaeem1/nutrition_project/logs/mmd_%j.out
#SBATCH --error=/work/alzaeem1/nutrition_project/logs/mmd_%j.err

set -e

cd /work/alzaeem1/nutrition_project

/home/alzaeem1/.conda/envs/nutritionml/bin/python \
    code/adapt_mmd.py \
    --model "$1"
