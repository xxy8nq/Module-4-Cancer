## Project Snapshot
- This repository is a student cancer data analysis project.
- Core work lives in `code/Module 4-Sullivan-Janousek.ipynb` (analysis notebook) and `code/example_EDA.py` (working Python example).
- Raw input data lives in `data/`; the main files are `TRAINING_SET_GSE62944_subsample_log2TPM.csv` and `TRAINING_SET_GSE62944_metadata.csv`.

## What the code does
- `example_EDA.py` loads gene expression and metadata CSVs with `pandas.read_csv(index_col=0)`.
- The expression file is structured as genes on rows and sample IDs on columns.
- The metadata file is indexed by sample ID and contains columns like `cancer_type`, `gender`, and `age_at_diagnosis`.
- Analysis follows this pattern:
  - subset metadata by `cancer_type`
  - use the matching sample IDs to subset expression columns
  - subset genes with `.loc[gene_list]`
  - transpose expression data (`.T`) before merging with metadata
  - plot with `seaborn` and `matplotlib`

## Local conventions
- Use relative data paths for new code, not absolute user-specific paths.
- Keep notebooks as the primary narrative vehicle; code scripts should support reproducible analysis and examples.
- The project currently has no CI/build/test scripts; the standard workflow is:
  - open `code/Module 4-Sullivan-Janousek.ipynb` in Jupyter/VS Code
  - run `python code/example_EDA.py` from the repo root

## Important details for edits
- Do not assume any hidden service architecture or backend.
- Focus on data wrangling and exploratory analysis, not on adding API/web service layers.
- If adding new scripts, place them under `code/` and mirror the existing pandas/plotting style.
- When merging expression and metadata, sample IDs are the join key; use the transposed expression frame so rows represent samples.

## Areas with missing implementation
- The notebook is mostly template text and sections to fill in; it does not contain a completed analysis pipeline.
- There is no package manifest (`requirements.txt`, `environment.yml`), so keep dependency changes minimal and explicit.

## Helpful file references
- `code/example_EDA.py` — concrete example of dataset loading, filtering, merging, and plotting
- `data/TRAINING_SET_GSE62944_subsample_log2TPM.csv` — expression matrix
- `data/TRAINING_SET_GSE62944_metadata.csv` — sample metadata

## If you are unsure
- Check the example script first for the expected data layout.
- Preserve the notebook structure and use the notebook for documentation of analysis decisions.
- Ask the user before introducing a new package or project-wide build/test workflow.
