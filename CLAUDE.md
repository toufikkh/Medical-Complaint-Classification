# CLAUDE.md — Medical Complaint Classification

Read `PRD.md` and `Medical_Complaint_Classification_Project.md` before doing anything. The original spec wins on any conflict.

## What this project is
A course graduation project with a hard deadline (< 24h). It classifies a patient's text complaint into a **medical category** (specialty) and shows the **top-3 categories with probabilities**. It is NOT disease prediction and NOT diagnosis.

Stack: Python, pandas, scikit-learn, matplotlib/seaborn, joblib, Streamlit, Jupyter.

## Golden rules
1. **Keep it simple, real, measurable, explainable.** A clean TF-IDF + Logistic Regression project beats a complicated one.
2. **Target is the category, never the disease.** Disease labels are only used to derive categories through one documented mapping dict.
3. **Inspect the dataset before modelling.** Do not trust a dataset by its name. Report: rows, columns, text/label columns, classes, distribution, missing values, duplicates, sample texts.
4. **No data leakage.** Fit TF-IDF on training data only. Use `sklearn.pipeline.Pipeline` (TfidfVectorizer → LogisticRegression). Drop duplicates before splitting.
5. **Do NOT remove stopwords** or negations ("no", "not", "without"). Preprocessing is conservative: lowercase, trim whitespace, handle missing, drop exact duplicates.
6. **Never present the system as diagnosis, triage, or clinically validated.** The disclaimer must appear in the app and README.
7. **Do not build anything on the non-goals list:** BERT, RAG, LLM chatbot, voice, databases, auth, deployment, scraping, Random Forest "for variety". The only API allowed is the minimal inference service `main.py`.
8. **Don't invent results.** Every number reported must come from actually running the code. If accuracy is very high, say the dataset is small/clean and flag it as a limitation.
9. **Ask before expanding scope.** If something isn't needed for the core deliverable, skip it.

## Modelling spec
- Split: 80/20, `stratify=y`, `random_state=42`.
- Vectorizer: `TfidfVectorizer(ngram_range=(1, 2))` (tune `min_df`/`max_features` only if needed).
- Model: `LogisticRegression(max_iter=1000)`; try `class_weight="balanced"` if classes are imbalanced.
- Baseline: `DummyClassifier(strategy="most_frequent")`.
- Optional (only if time): `LinearSVC` for comparison.
- Metrics: accuracy, `classification_report`, macro-F1, confusion matrix, **top-3 accuracy**.
- Error analysis: ~10 misclassified examples with a short explanation.
- Save: `joblib.dump(pipeline, "models/pipeline.joblib")` (contains the fitted vectorizer and classifier).

## Streamlit app (`app.py`)
- Loads `models/pipeline.joblib`; never retrains.
- Uses the same preprocessing function as training (keep it in one shared place).
- Uses `predict_proba`, shows the top 3 categories as percentages, sorted.
- Shows a "low confidence" note if the top probability is below ~50%.
- Handles empty input.
- Always displays the disclaimer.

## FastAPI service (`main.py`)
- Inference only: `GET /health`, `GET /categories`, `POST /predict` (`{"text", "top_k"}`).
- Loads `models/pipeline.joblib` at start-up; returns 503 with a helpful message if missing.
- `clean_text` must stay identical to the notebook and `app.py`.
- Response includes `low_confidence` and the disclaimer. No database, no auth.

## Project structure
```
data/            raw files (data/processed/ for cleaned data)
notebooks/analysis.ipynb
models/pipeline.joblib, metadata.json
reports/         figures
docs/EXPLANATIONS.md
app.py           Streamlit
main.py          FastAPI
requirements.txt
README.md
PRD.md
CLAUDE.md
```
Keep it flat; don't add packages, configs, or frameworks that aren't needed.

## Commands
```bash
pip install -r requirements.txt
jupyter notebook notebooks/analysis.ipynb
streamlit run app.py
uvicorn main:app --reload     # API docs at /docs
```

## Working style
- Work in small steps in this order: inspect data → clean + map categories → EDA → split → train → evaluate → error analysis → save → app → README.
- After each step, show the key output (shapes, counts, metrics) so it can be checked.
- Fixed random seeds; the notebook must run top-to-bottom.
- Short comments that explain *why*; use plain language, since the author must be able to explain every line.
- When unsure about a labeling decision (disease → category), flag it and propose an option rather than silently choosing.

## Definition of done
- Notebook runs cleanly end to end.
- Dataset inspection and category mapping documented.
- Baseline + model metrics + confusion matrix + top-3 accuracy reported.
- App returns top-3 categories with percentages.
- README includes: overview, dataset, mapping, results, how to run, limitations, disclaimer.
- Slides outline follows the original spec (problem → solution → dataset → EDA → preprocessing → model → evaluation → app → limitations → future work).

## If time runs short, cut in this order
Linear SVM → text-length EDA → error-analysis depth → app polish.
Never cut: evaluation, disclaimer, working demo.
