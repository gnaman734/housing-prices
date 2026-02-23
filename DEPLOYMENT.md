# Public Deployment Guide (Milestone 1)

This guide helps you satisfy the requirement that localhost-only demos are not accepted.

## Option A: Streamlit Community Cloud (Recommended)

## 1) Push project to GitHub
```bash
cd "/Users/namangupta734/Documents/New project"
git init
git add .
git commit -m "Project 9 Milestone 1"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## 2) Deploy on Streamlit Cloud
1. Open https://share.streamlit.io/
2. Sign in with GitHub.
3. Click "New app".
4. Select your repo and branch (`main`).
5. Set main file path as `app.py`.
6. Deploy.

## 3) Verify live app
- Confirm training CSV upload works.
- Confirm prediction CSV upload + download works.
- Save the public URL for submission.

## Option B: Hugging Face Spaces (Streamlit)
1. Create a new Space at https://huggingface.co/new-space
2. Choose SDK = Streamlit.
3. Upload repository files.
4. Ensure `requirements.txt` and `app.py` are present.
5. Wait for build, then share public Space URL.

## Required Submission Artifacts
- Public app URL.
- GitHub repository URL.
- Report PDF (compiled from `report/main.tex`).
- Notebook file (`notebooks/notebook778e406a1b.ipynb`).
