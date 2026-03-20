# asr_dialect_classification

# 1) README.md

# Reproducibility Package – PB–SP Prosodic-Acoustic Dialect Classification

This repository contains the minimal materials required to reproduce the inferential and predictive analyses reported in the manuscript:

Manuscript title:
“The influence of prosodic-acoustic features in automatic speech recognition of two inland Brazilian dialects: A pilot study”

## Repository structure

.
├── README.md
├── CODEBOOK.md
├── requirements.txt
├── environment_R.txt
├── data/
│   ├── asr_dataset.csv
│   ├── lmem_dataset.csv
│   ├── train_test_split_ids.csv
│   └── corpus_materials/
│       ├── segmentation_guidelines.txt
│       └── metadata_template.csv
├── scripts/
│   ├── run_lmem_analysis.R
│   └── run_asr_classification.py
├── outputs/
│   ├── lmem_results.csv
│   ├── classification_report.txt
│   ├── confusion_matrices.csv
│   └── feature_importance.csv
└── ACCESS_RULES.md

## Files

### data/asr_dataset.csv
Processed dataset used in machine-learning classification.
Required columns:
- `sample_id`
- `dialect`
- the nine statistically significant acoustic variables:
  `f0sd`, `f0SAQ`, `df0mean_pos`, `df0sd_pos`,
  `sl_LTAS_alpha`, `cvint`,
  `pause_sd`, `pause_meandur`, `pause_rate`

### data/lmem_dataset.csv
Processed dataset used in mixed-effects modeling.
Required columns:
- `sample_id`
- `speaker`
- `dialect`
- the same nine acoustic variables listed above

### data/train_test_split_ids.csv
Exact record of the train–test partition used in the ASR classification stage.
Generated automatically by the Python script with:
- `sample_id`
- `dialect`
- `set` (train/test)

### data/corpus_materials/
Contains the shareable corpus materials required to reconstruct the processing pipeline:
- podcast script;
- segmentation guidelines;
- metadata template for participants and files.

**Raw audio is not public** for ethical reasons. See `ACCESS_RULES.md`.

## Execution order

### Step 1 – Inferential analysis (R)
Run:

```bash
Rscript scripts/run_lmem_analysis.R

This script:

	1. reads data/lmem_dataset.csv;

	2. fits one linear mixed-effects model per acoustic variable;

	2. outputs outputs/lmem_results.csv.

### Step 2 – Predictive analysis (Python)

Run:

python scripts/run_asr_classification.py

This script:

	1. reads data/asr_dataset.csv;

	2. normalizes predictors;

	3. recreates the exact train–test split using random_state = 1;

	4. trains six classifiers;

outputs:
	
	- outputs/classification_report.txt

	- outputs/confusion_matrices.csv

	- outputs/feature_importance.csv

	- data/train_test_split_ids.csv

Random seed and split

The classification stage uses:

	1. random_state = 1

	2. test_size = 0.20

	3. The exact partition is saved in data/train_test_split_ids.csv.

Computational environment

	See:

		1. requirements.txt for Python packages

		2. environment_R.txt for R packages and session information

## Ethical limitations

	## Raw audio recordings are restricted because they contain identifiable voice data and are part of an ongoing speech database. 
	
	=======
	
	# Computational Environment and Reproducibility Settings

## Software

- Python version: 3.11
- R version: 4.5.0

## Python libraries

- numpy >= 1.24
- pandas >= 2.0
- scikit-learn >= 1.3
- matplotlib >= 3.7

## R packages

- lme4
- lmerTest
- performance
- dplyr
- readr

## Parameters

- Random seed: 1
- Train-test split: 80% training / 20% testing
- Stratification: by dialect (PB, SP)

## Model configuration

The machine learning models were trained using GridSearchCV with 5-fold cross-validation to optimize hyperparameters.

## Notes

All analyses were conducted on a standard desktop computing environment. No GPU acceleration was required.
