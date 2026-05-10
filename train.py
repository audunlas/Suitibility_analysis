"""CHAP train entry point — serializes model config to disk."""

import json
import os
import sys

import joblib

from suitability.model import SuitabilityModel


def main():
    if len(sys.argv) < 3:
        print("Usage: python train.py <train_data_csv> <model_output_path> [config.json]")
        sys.exit(1)

    config_path = sys.argv[3] if len(sys.argv) > 3 else "configs/chap_malaria.json"
    with open(config_path) as f:
        model = SuitabilityModel.from_dict(json.load(f))

    model_dir = os.path.dirname(sys.argv[2])
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model.to_dict(), sys.argv[2])
    print(f"Model config saved to {sys.argv[2]}")


if __name__ == "__main__":
    main()
