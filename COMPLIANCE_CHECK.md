# Project 9 Compliance Check (Milestone 1)

Date checked: 2026-02-23
Repository: https://github.com/gnaman734/housing-prices

## Constraints and Requirements
- Paid APIs not permitted: PASS
  - Uses open-source stack only (`scikit-learn`, `pandas`, `numpy`, `streamlit`).
- Free-tier APIs/open-source models only: PASS
  - No paid API integration present.
- User interface mandatory: PASS
  - Streamlit UI in `app.py` with training and prediction tabs.
- Final app must be publicly hosted: PENDING USER ACTION
  - Deployment guide is ready in `DEPLOYMENT.md`; public URL not yet added.
- Localhost-only demos not accepted: PENDING USER ACTION
  - Satisfied after publishing to Streamlit Cloud/HF Spaces.

## Milestone 1 Functional Requirements
- Data preprocessing and feature engineering: PASS
  - Implemented in `src/modeling.py`.
- Predict property prices/categories: PASS
  - Regression price prediction implemented.
- Identify key price-driving factors: PASS
  - Feature importance displayed in UI.
- Display predictions through basic UI: PASS
  - CSV upload + batch predictions + CSV download in `app.py`.

## Milestone 1 Technical Requirements
- Supervised models (LR/DT/RF): PASS
  - All three supported and selectable.
- Optional neural network: NOT REQUIRED
- Metrics MAE/RMSE/R2: PASS
  - Computed and displayed/saved.

## Milestone 1 UI Requirements
- Upload property data: PASS
- Display predicted prices/ranges: PASS
- Show explanation of price-driving features: PASS

## Milestone 1 Deliverables Check
- Problem understanding/use-case description: PASS
  - `report/main.tex` Introduction.
- Input-output specification: PASS
  - Added explicitly in report.
- System architecture diagram (ML pipeline): PASS
  - Added in report as architecture block.
- Working local app with basic UI: PASS
  - `app.py` implemented.
- Brief model performance evaluation: PASS
  - Results table in report.

## File Format/Template Alignment
- IEEE-style report format used: PASS (`report/main.tex`)
- Bibliography present: PASS (`report/references.bib`)
- Appendix with GitHub repository info: PASS

## Remaining Final Steps
1. Deploy app publicly and record URL.
2. Add public app URL in report (suggested in Appendix).
3. Compile PDF from `report/main.tex`.
