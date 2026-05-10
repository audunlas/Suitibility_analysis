"""CHAP report entry point — generates PDF from suitability analysis."""

import json
import sys

from suitability.model import SuitabilityModel
from analysis.pipeline import load_and_clean, run_all_analyses
from visualize import generate_pdf

CONFIG_PATH = "configs/chap_malaria.json"


def main():
    if len(sys.argv) != 4:
        print("Usage: python report.py <model> <historic_data_csv> <out_file.pdf>")
        sys.exit(1)

    _, historic_data_path, out_file = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(CONFIG_PATH) as f:
        model = SuitabilityModel.from_dict(json.load(f))

    df, outcome_col = load_and_clean(historic_data_path, model)
    all_results = run_all_analyses(df, model, outcome_col)
    generate_pdf(all_results, model, df, out_file)


if __name__ == "__main__":
    main()
