# PRD — Medical Complaint Classification (NLP + ML)

**Status:** v1.1 (implemented) · **Type:** Course graduation project · **Time budget:** < 24h total (≈ 6–8h of actual work)
**Source of truth for scope:** `Medical_Complaint_Classification_Project.md` (the original spec). If this PRD and the spec conflict, the spec wins.

---

## 1. Summary

Build a simple NLP system that takes a patient's free-text complaint and returns the **most likely medical categories** (specialties) with a probability for each, e.g.:

```
"I have red patches on my skin and severe itching."
→ Dermatology 71% · Allergy 15% · Infectious Disease 7%
```

This is **category classification, NOT disease prediction and NOT diagnosis.**

## 2. Goals

1. Deliver a complete, working, explainable MVP: data → cleaning → EDA → TF-IDF → Logistic Regression → evaluation → Streamlit app.
2. Show top-N (N=3) category predictions with probabilities, not a single answer.
3. Evaluate honestly (not accuracy only) and discuss limitations.
4. Be defensible by a beginner data-science student.

## 3. Non-Goals (do NOT build)

Disease prediction · diagnosis · urgency/emergency triage · BERT/transformers · RAG · LLM chatbot · knowledge base · voice · databases · auth · deployment infrastructure · web scraping. (A single minimal inference API, FR13, is the only API allowed.) These may appear only under "Future Work".

## 4. Research Question

> Can a traditional NLP pipeline (TF-IDF + supervised ML) accurately classify short medical complaints into predefined medical categories?

## 5. Dataset & Labels

### 5.1 Selection (inspect BEFORE modelling)
- **Primary candidate (links):** Hugging Face <https://huggingface.co/datasets/gretelai/symptom_to_diagnosis> · Kaggle <https://www.kaggle.com/datasets/niyarrbarman/symptom2disease>.
- **Description:** Symptom2Disease — natural-language symptom text, ~1,200 rows, ~22–24 disease labels, ~50 texts per disease. Available on Hugging Face (`gretelai/symptom_to_diagnosis`, train/test jsonl, Apache-2.0) and Kaggle (`niyarrbarman/symptom2disease`). The two versions may differ in class count; confirm which one is used.
- **Fallback:** a subset of the large Kaggle `abhishekgodara/symptoms_to_diseases` processed NLP file (keep only the most frequent diseases). Use only if the primary is unusable.
- Do **not** pick a dataset by name alone. Complete the inspection checklist (§5.2) and record results in the notebook and README.

### 5.2 Inspection checklist (must be answered)
Rows · column names · text column · label column · number of classes · class distribution · missing values · duplicates (exact and near) · does the text read like complaints/symptoms · are labels meaningful · is the dataset usable.

### 5.3 Category labels (disease → category mapping)
The dataset is labeled by disease; the project target is the **medical category**. Create the category label with one explicit mapping dictionary in code (`src` or notebook), kept in one place and documented in the README.

**Draft mapping — verify against the actual label set after inspection:**

| Category | Diseases (draft) |
|---|---|
| Dermatology | psoriasis, impetigo, fungal infection, acne (if present) |
| Infectious Disease | dengue, malaria, typhoid, chicken pox |
| Respiratory | pneumonia, bronchial asthma, common cold |
| Gastroenterology | GERD, peptic ulcer disease, jaundice |
| Cardiovascular | hypertension, varicose veins |
| Neurology | migraine |
| Musculoskeletal | arthritis, cervical spondylosis |
| Endocrinology | diabetes |
| Urology | urinary tract infection |
| Allergy | allergy, drug reaction |

Rules:
- Every disease must map to exactly one category; no unmapped rows may silently disappear (assert this).
- Ambiguous cases (e.g., chicken pox, drug reaction, jaundice) are judgment calls: pick one, document it, and mention it in Limitations.
- If a category ends up with very few examples, merge it into a sensible neighbour (e.g., Endocrinology/Urology into "Other/General") and document it.
- The mapping is **not clinically validated**. State this in the report.

## 6. Functional Requirements

| ID | Requirement |
|---|---|
| FR1 | Load data, inspect it, and report the checklist in §5.2 |
| FR2 | Clean conservatively: handle missing values, drop exact duplicates, lowercase, trim whitespace. **Do NOT remove stopwords** (keep "no", "not", "without") |
| FR3 | Apply the disease→category mapping and verify no rows are lost unintentionally |
| FR4 | EDA: dataset size, #categories, class-distribution bar chart, missing/duplicates, text-length distribution (optional). Mention imbalance if present |
| FR5 | Stratified 80/20 train/test split with a fixed `random_state` |
| FR6 | Pipeline: `TfidfVectorizer(ngram_range=(1,2))` → `LogisticRegression` (multinomial). TF-IDF fitted on training data only (use `sklearn.pipeline.Pipeline` so leakage is impossible) |
| FR7 | Baseline: majority-class `DummyClassifier` for context |
| FR8 | Optional: Linear SVM comparison if time permits. No Random Forest or other models |
| FR9 | Evaluation: accuracy, per-class precision/recall/F1 (classification report), macro-F1, confusion matrix, **top-3 accuracy** |
| FR10 | Error analysis: inspect ~10 misclassified examples and explain likely overlaps |
| FR11 | Save the fitted pipeline with `joblib` (contains vectorizer + classifier) to `models/` |
| FR12 | Streamlit app (see §7) |
| FR13 | Minimal FastAPI service `main.py` (inference only): `GET /health`, `GET /categories`, `POST /predict` returning top-k categories with probabilities. No database, no auth. Added by explicit request of the project owner; it must stay minimal |

## 7. Streamlit App

**Input:** text area for the complaint, "Classify" button.
**Output:** top-3 categories with probabilities (sorted, shown as percentages with a bar or table).
**Behaviours:**
- Load the saved pipeline; never retrain in the app.
- Apply the same preprocessing as training (shared function).
- Use `predict_proba`; show the top 3.
- If the top probability is below ~50%, show a "low confidence" notice.
- Handle empty input gracefully.
- Always show the disclaimer: *"This application is for educational/research purposes. It classifies complaints into predefined categories and does not provide medical diagnosis or emergency advice."*

Note: Logistic Regression probabilities are model scores, not calibrated clinical probabilities. Say so in the README.

## 8. Success Criteria

The MVP is done when:
1. The notebook runs top-to-bottom without errors on a clean environment.
2. Data inspection results and the mapping are documented.
3. Evaluation includes: baseline, accuracy, classification report, macro-F1, confusion matrix, top-3 accuracy.
4. The Streamlit app and the FastAPI `/predict` endpoint return top-3 categories with percentages for a typed complaint.
5. The README states limitations and the non-diagnosis disclaimer.
6. Results are reported honestly. Very high accuracy on a small, clean dataset must be discussed as a limitation, not presented as real-world performance.

No numeric accuracy target is promised in advance; the result is whatever the data supports.

## 9. Deliverables

```
medical-complaint-classifier/
├── data/               # raw + processed data
├── notebooks/analysis.ipynb
├── models/             # saved pipeline (.joblib)
├── app.py              # Streamlit
├── main.py             # FastAPI (inference only)
├── reports/            # figures saved by the notebook
├── docs/EXPLANATIONS.md
├── requirements.txt
├── README.md
├── PRD.md
└── CLAUDE.md
```
Plus a short presentation following the structure in the original spec (problem → solution → dataset → EDA → preprocessing → model → evaluation → app → limitations → future work).

## 10. Plan (24h window, ≈ 6–8h work)

| Block | Task | Time |
|---|---|---|
| 1 | Download + inspect dataset, decide mapping | 1–1.5h |
| 2 | Cleaning + category mapping + EDA | 1h |
| 3 | Split, pipeline, baseline, evaluation | 1.5h |
| 4 | Error analysis, (optional SVM) | 0.5h |
| 5 | Save model + Streamlit app | 1h |
| 6 | README + slides | 1.5h |

**Cut order if time runs short:** Linear SVM → text-length EDA → error analysis depth → app polish. Never cut: evaluation, disclaimer, working demo.

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Dataset unsuitable | Cap search at ~2h; use the primary candidate and document its limits |
| Category mapping is subjective | One documented dictionary; discuss ambiguity in Limitations |
| Small/synthetic data → inflated accuracy | Baseline, stratified split, dedupe, honest discussion |
| Duplicate texts across train/test | Drop duplicates before splitting |
| Overlapping categories (e.g., Respiratory vs Infectious) | Top-3 output + confusion matrix + error analysis |
| Scope creep | Follow §3 strictly |

## 12. Safety & Limitations (must appear in README, app, and slides)

Not a diagnosis system, not a doctor replacement, not emergency triage, not clinically validated. Quality depends on dataset quality; the data is small, likely synthetic or not clinically validated, and category labels come from a manual mapping.

## 13. Known scope gap

The original spec's example "Tooth pain → Dentistry" cannot be supported by Symptom2Disease (no dental class). The app must not claim dental support; out-of-scope complaints are handled only by the low-confidence warning.

## 14. Future Work (mention only)

Structured inputs (age, duration) · urgency classification · transformer models (BERT/BioBERT/ClinicalBERT) · speech interface · real clinical text · probability calibration.
