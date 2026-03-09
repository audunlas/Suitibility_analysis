"""Run suitability analysis for Lao PDR malaria data."""

import os
import sys

# Ensure imports resolve relative to this script's directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import train
from configs.lao_malaria import LAO_MALARIA

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data/chap_LAO_admin1_monthly.csv"
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "output/lao/model.pkl"
)

if __name__ == "__main__":
    print(f"Data:   {DATA_PATH}")
    print(f"Output: {OUTPUT_PATH}\n")
    train(DATA_PATH, OUTPUT_PATH, model=LAO_MALARIA)
