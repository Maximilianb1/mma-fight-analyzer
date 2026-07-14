# Report source

This folder contains the Overleaf-ready source for the final project report.

## Overleaf

Upload the complete `report/` folder to a blank Overleaf project and set `main.tex` as the main document. All generated figures are already included, so no Python execution is required on Overleaf.

Before submission, replace the highlighted placeholder in Ethics Statement section 3a with the students' independent reflection. The course explicitly prohibits using an LLM for that paragraph.

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
