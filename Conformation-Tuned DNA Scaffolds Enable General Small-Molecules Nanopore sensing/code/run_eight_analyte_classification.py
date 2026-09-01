from __future__ import annotations

import argparse
from pathlib import Path

from contour_ml_common import EIGHT_ANALYTE_CLASSES, build_eight_analyte_models, evaluate_holdout, load_feature_table, save_holdout_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the 13 reported classifiers for eight CONTOUR-DNA analytes.")
    parser.add_argument("--input", type=Path, required=True, help="External feature workbook; no data are included in this release.")
    parser.add_argument("--sheet", default="Expanded_Data", help="Workbook sheet containing the feature table.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Local directory for generated metrics and manifests.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state used for the reported split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, X, y = load_feature_table(args.input, EIGHT_ANALYTE_CLASSES, args.sheet)
    metrics, predictions = evaluate_holdout(X, y, build_eight_analyte_models(args.random_state), args.random_state)
    save_holdout_run(args.output_dir, metrics, predictions, EIGHT_ANALYTE_CLASSES, args.random_state)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()


