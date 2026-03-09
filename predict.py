"""CHAP predict entry point.

Minimal stub to satisfy CHAP's train/predict interface.
Outputs a constant placeholder — the real analysis is in train.py.
"""

import sys
import os

import pandas as pd


def predict(
    model_path: str,
    historic_data_path: str,
    future_data_path: str,
    out_file: str,
):
    """Write CHAP-format output with placeholder values."""
    df = pd.read_csv(future_data_path)

    output = pd.DataFrame({
        "time_period": df["time_period"],
        "location": df["location"],
        "sample_0": 0,
    })

    os.makedirs(os.path.dirname(out_file) if os.path.dirname(out_file) else ".", exist_ok=True)
    output.to_csv(out_file, index=False)
    print(f"Predictions saved to {out_file} ({len(output)} rows)")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python predict.py <model_path> <historic_data> <future_data> <out_file>")
        sys.exit(1)
    predict(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
