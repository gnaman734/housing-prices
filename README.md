# Project 9: Intelligent Property Price Prediction and Agentic Real Estate Advisory System

This repository contains a complete Milestone 1 implementation plus a report-ready structure based on your IEEE template.

## Included Assets
- Streamlit app: `app.py`
- Modular ML pipeline: `src/modeling.py`, `src/io_utils.py`
- Kaggle notebook copy: `notebooks/notebook778e406a1b.ipynb`
- Notebook-equivalent runnable baseline script: `src/notebook_baseline.py`
- IEEE report source: `report/main.tex`, `report/references.bib`

## Environment Setup
```bash
cd "/Users/namangupta734/Documents/New project"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the UI
```bash
streamlit run app.py
```

## Run Notebook Baseline Pipeline (CLI)
Use this with Kaggle `Housing.csv` path downloaded locally.
```bash
python src/notebook_baseline.py --data /path/to/Housing.csv --outdir results
```
Outputs:
- `results/notebook_baseline_metrics.json`
- `results/best_house_price_model.pkl`

## Compile Report
```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Public Hosting (Free Tier)
- Streamlit Community Cloud
- Hugging Face Spaces (Streamlit SDK)

Both satisfy the non-localhost requirement when deployed publicly.

## Submission Checklist
- Train and test the app using your target dataset (`Housing.csv` or course dataset).
- Capture final metrics and keep them consistent with `report/main.tex`.
- Deploy publicly (Streamlit Cloud or Hugging Face Spaces).
- Update GitHub link in `report/main.tex` appendix.
- Export PDF from `report/main.tex` and submit code + report.

See detailed deployment steps in `DEPLOYMENT.md`.
