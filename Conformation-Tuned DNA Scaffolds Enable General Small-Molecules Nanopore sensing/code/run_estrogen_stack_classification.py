from __future__ import annotations

import argparse
from pathlib import Path

from contour_ml_common import ESTROGEN_CLASSES, evaluate_stack_oof, load_feature_table, save_oof_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the CONTOUR-DNA Stack classifier for NIL, EE2, and MES.")
    parser.add_argument("--input", type=Path, required=True, help="External feature workbook; no data are included in this release.")
    parser.add_argument("--sheet", default="Expanded_Data", help="Workbook sheet containing the feature table.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Local directory for generated metrics and manifests.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state used for the outer cross-validation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, X, y = load_feature_table(args.input, ESTROGEN_CLASSES, args.sheet)
    metrics, predictions, probabilities, folds = evaluate_stack_oof(X, y, args.random_state)
    save_oof_run(args.output_dir, metrics, y, predictions, probabilities, ESTROGEN_CLASSES, folds, args.random_state)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()


