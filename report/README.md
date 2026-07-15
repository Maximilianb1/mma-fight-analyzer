# Report source

This folder contains the Overleaf-ready source for the final project report.

## Overleaf

Upload the complete `report/` folder to a blank Overleaf project and set `main.tex` as the main document. All generated figures are already included, so no Python execution is required on Overleaf.

Ethics Statement section 3a contains the students' independently written responses. They were inserted verbatim because the course explicitly prohibits using an LLM to write or revise that paragraph; both students should verify the wording before submission.

## Local compilation

```bash
cd report
latexmk -pdf main.tex
```

From the repository root, regenerate the quantitative figures from the checked-in report data with:

```bash
python report/scripts/generate_figures.py
```

The numerical tables under `report/data/` were copied from the final saved experiment artifacts and independently verified against the prediction files.

The Grad-CAM panels under `report/figures/` are qualitative diagnostic outputs. They are presented as selected examples rather than a quantitative localization evaluation; the report states their interpretation limits explicitly.

To generate an audited EigenGradCAM analysis with the released phase checkpoint, run:

```bash
python scripts/download_data.py
python scripts/preprocess.py
python scripts/download_models.py
python report/scripts/generate_explainability.py
```

The generator matches validation-time resizing, targets the displayed phase explicitly, and writes a JSON manifest with the checkpoint, candidate split, and selected clips. Its default `development` split contains training examples for the final deployment model; use `--candidate-split holdout` for a strictly unseen qualitative analysis, which may not contain a correct example for every class.
