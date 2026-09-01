# Conformation-Tuned DNA Scaffolds Enable General Small-Molecules Nanopore sensing

This folder contains the code-only release for the machine-learning analyses
reported in the CONTOUR-DNA manuscript. No experimental data, source tables,
figures, trained model files, or local dependency caches are included.

The scripts read an externally supplied workbook. The published feature schema
uses the names reported in the manuscript and Supplementary Information:

```text
I/I0, ΔSD, Skew, Median I/I0, Dwell time (ms), Kurt, IQR I/I0, CV
```

For compatibility with the pre-release local workbook only, the loader accepts
`English abbreviation`, `RMS (pA)`, `Dwell Time (ms)`, and `Kurtosis` as legacy
names for the published `Analyte`, `ΔSD`, `Dwell time (ms)`, and `Kurt` fields.
It also maps the old class labels `DP2`, `IDP2`, `DP3`, and `IDP3` to the
published `Mal`, `IsoMal`, `Mal3`, and `IsoMal3` labels. The legacy workbook
itself is not part of this release.

## Included analyses

- `code/contour_ml_common.py`: shared feature loading, model definitions, and
  evaluation routines.
- `code/run_eight_analyte_classification.py`: the 13-model comparison and the
  Stack classifier for Mal, IsoMal, Mal3, IsoMal3, 3-DG, 5-AMF, 5-EMF, and 5-HMF.
- `code/run_estrogen_stack_classification.py`: the Stack classifier for NIL,
  EE2, and MES using the generalizability assay class order.

The eight-analyte comparison uses the reported stratified 3:1
training-validation split and random state 42. The Stack combines SVM-RBF,
Random Forest, XGBoost, and GaussianNB, followed by the ordinary Logistic
Regression meta learner specified in Supplementary Table 8 (`max_iter=2000`,
default logistic solver/penalty, `class_weight="balanced"`, `cv=3`,
`stack_method="predict_proba"`).

## Usage

Tested with Python 3.12 and the versions pinned in `requirements.txt`.

```bash
pip install -r requirements.txt
python code/run_eight_analyte_classification.py \
  --input /path/to/feature_workbook.xlsx \
  --sheet Expanded_Data \
  --output-dir /path/to/local_outputs

python code/run_estrogen_stack_classification.py \
  --input /path/to/estrogen_feature_workbook.xlsx \
  --sheet Expanded_Data \
  --output-dir /path/to/local_outputs/estrogens
```

Generated metrics, predictions, confusion matrices, and manifests are written
only to the user-specified output directory and are intentionally excluded from
the repository.

