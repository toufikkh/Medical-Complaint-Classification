# 🩺 Medical Complaint Classification (NLP + Machine Learning)

A small, explainable NLP project: given a patient's **free-text complaint**, the system returns the **top-3 most likely medical categories** (e.g. Dermatology, Respiratory) with a **probability for each**.

```
"I have red patches on my skin and severe itching."
        |
   text cleaning  ->  TF-IDF  ->  Logistic Regression
        |
   Dermatology 71%  ·  Allergy 15%  ·  Infectious Disease 7%
```

> ⚠️ **This is NOT a medical diagnosis system.** It does not identify diseases, is not an emergency/triage tool, and is not clinically validated. It is an educational/research prototype that classifies text into predefined categories.

---

## 1. What is in this project

| File / folder | Purpose |
|---|---|
| `notebooks/analysis.ipynb` | **Everything about data and training**: load, inspect, clean, map categories, EDA, split, train, evaluate, error analysis, save model |
| `app.py` | **Streamlit** web interface (type a complaint, see top-3 categories) |
| `main.py` | **FastAPI** service (`POST /predict`) for programs that want to call the model |
| `models/` | Created by the notebook: `pipeline.joblib` (the trained model) and `metadata.json` |
| `data/` | The dataset files (downloaded automatically by the notebook, or placed manually) |
| `reports/` | Figures saved by the notebook (class distribution, confusion matrix) for your slides |
| `docs/EXPLANATIONS.md` | Concepts explained simply + questions you may be asked + presentation outline |
| `PRD.md` / `CLAUDE.md` | Project specification and working rules |
| `requirements.txt` | Python dependencies |

> The trained model is **not** included in this zip on purpose: you generate it by running the notebook (about a minute), so every number you present comes from your own run on the real dataset.

---

## 2. Dataset

**Primary dataset (recommended): Symptom2Disease-style data.** Natural-language symptom descriptions, each labeled with a disease.

| Source | Link |
|---|---|
| Hugging Face (used by the notebook) | <https://huggingface.co/datasets/gretelai/symptom_to_diagnosis> |
| Direct files | `https://huggingface.co/datasets/gretelai/symptom_to_diagnosis/resolve/main/train.jsonl` and `.../test.jsonl` |
| Kaggle version (alternative) | <https://www.kaggle.com/datasets/niyarrbarman/symptom2disease> |

- The notebook **downloads the Hugging Face files automatically** the first time (needs internet).
- Manual option: download the files yourself and put the `.jsonl` or `.csv` files in the `data/` folder. The notebook detects the text and label columns automatically, so either version works.
- The Hugging Face and Kaggle versions may not have exactly the same diseases. The notebook stops with a clear message if a disease is not in the category mapping, and you add one line to fix it.

**Why a mapping is needed.** The dataset labels are **diseases**, but this project predicts **medical categories**. Predicting about 10 categories needs far less data and is much less risky than predicting 20+ diseases. The mapping (section 6) is our own, **not clinically validated**.

**Honest dataset warnings** (also in the notebook's limitations):
- It is small (about a thousand short texts).
- The texts look clean and uniform, probably synthetic or not clinically validated. Real patient messages are messier, so real-world accuracy would be lower.
- It contains **no dental complaints**: the model cannot return "Dentistry". A toothache complaint will be forced into some other category (usually with low confidence).

---

## 3. Quick start

### 3.1 Install

```bash
# (recommended) create an isolated environment
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

### 3.2 Train the model (run the notebook first!)

```bash
jupyter notebook notebooks/analysis.ipynb
```
Then **Kernel -> Restart & Run All**. At the end you will have:
- `models/pipeline.joblib` and `models/metadata.json`
- `reports/class_distribution.png`, `reports/text_length.png`, `reports/confusion_matrix.png`
- `data/processed/complaints_clean.csv`

Read the printed results and fill in the *"Inspection conclusions"* and *"Error analysis notes"* markdown cells.

### 3.3 Run the Streamlit app

```bash
streamlit run app.py
```
Opens at <http://localhost:8501>.

### 3.4 Run the API

```bash
uvicorn main:app --reload
```
Interactive docs at <http://127.0.0.1:8000/docs>.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I have red patches on my skin and severe itching.", "top_k": 3}'
```

Example response (values illustrative):

```json
{
  "input_text": "I have red patches on my skin and severe itching.",
  "predictions": [
    {"category": "Dermatology", "probability": 0.71, "percentage": 71.0},
    {"category": "Allergy", "probability": 0.15, "percentage": 15.0},
    {"category": "Infectious Disease", "probability": 0.07, "percentage": 7.0}
  ],
  "low_confidence": false,
  "disclaimer": "Educational/research prototype. ..."
}
```

Other endpoints: `GET /health`, `GET /categories`, `GET /`.
If the model file is missing, `/predict` answers **503** with a message telling you to run the notebook.

---

## 4. How it works (step by step)

1. **Load and inspect** the data (11-point checklist: rows, columns, classes, distribution, missing values, duplicates, whether the text and labels make sense).
2. **Clean** conservatively: drop missing values, lowercase, remove odd characters, collapse spaces, drop duplicates. **Stop-words are kept** because "no", "not", "without" change medical meaning.
3. **Map** diseases to medical categories with one documented dictionary. Remove ambiguous texts (same text, two categories) and merge categories that are too small.
4. **EDA**: class-distribution bar chart and text-length histogram.
5. **Split** 80/20, stratified, fixed seed.
6. **Baseline**: always predict the most frequent category (to know what "better than nothing" means).
7. **Model**: `Pipeline(TF-IDF with 1-2 word n-grams -> Logistic Regression)`, tuned with 5-fold cross-validation **on the training set only** (`class_weight="balanced"`).
8. **Evaluate on the untouched test set**: accuracy, precision, recall, F1, macro-F1, confusion matrix, top-3 accuracy.
9. **Optional comparison** with a Linear SVM.
10. **Error analysis** and **top words per category** (explainability).
11. **Save** the whole pipeline with `joblib`.
12. **Serve** it in Streamlit (`app.py`) and FastAPI (`main.py`); both load the file and never retrain.

**No data leakage.** TF-IDF lives inside the scikit-learn `Pipeline`, so it only ever learns from training texts. Duplicates are removed before splitting.

**One file instead of two.** The original specification says to save the classifier and the vectorizer. We save them together as one `Pipeline` object, which is equivalent and prevents mismatching the two.

---

## 5. Evaluation (fill in with YOUR numbers)

Run the notebook, then copy the values into this table for your report and slides.

| Model | Accuracy | Macro-F1 | Top-3 accuracy |
|---|---|---|---|
| Baseline (most frequent) | _from notebook_ | _from notebook_ | - |
| TF-IDF + Logistic Regression | _from notebook_ | _from notebook_ | _from notebook_ |
| TF-IDF + Linear SVM (optional) | _from notebook_ | _from notebook_ | _from notebook_ |

**How to interpret them**
- Compare with the baseline first. A model must clearly beat it.
- **Accuracy** can hide weak categories, so read the per-category precision/recall and **macro-F1**.
- **Top-3 accuracy** is what the app shows: is the right category among the three displayed?
- If scores are very high (close to 100%), do **not** present that as real-world performance. Say that the dataset is small, clean and probably synthetic, and list it as a limitation.

Probabilities shown in the app are **model scores**, not calibrated medical risk.

---

## 6. Category mapping (draft, not clinically validated)

| Category | Diseases in the dataset |
|---|---|
| Dermatology | psoriasis, impetigo, fungal infection, (acne) |
| Infectious Disease | dengue, malaria, typhoid, chicken pox |
| Respiratory | pneumonia, bronchial asthma, common cold |
| Gastroenterology | GERD, peptic ulcer disease, jaundice, (dimorphic hemorrhoids) |
| Cardiovascular | hypertension, varicose veins |
| Neurology | migraine |
| Musculoskeletal | arthritis, cervical spondylosis |
| Endocrinology | diabetes |
| Urology | urinary tract infection |
| Allergy | allergy, drug reaction |

Diseases in parentheses exist only in some dataset versions. Judgment calls to mention in your report: *chicken pox* (Infectious vs Dermatology), *drug reaction* (Allergy vs Dermatology), *jaundice* (Gastroenterology vs other). You can change the mapping in section 5 of the notebook (`CATEGORY_MAP`), then re-run.

---

## 7. Limitations

1. Small dataset; scores can vary with another random split.
2. Probably synthetic/not clinically validated text; real complaints are noisier.
3. Disease-to-category mapping is manual and not clinically validated.
4. Closed set of categories: the model cannot say "I don't know" or "dentistry"; it can only warn through low confidence.
5. TF-IDF is a bag-of-words method; it does not truly understand sentences (negation is only partly captured through word pairs).
6. Probabilities are model scores, not medical risk.
7. Not a diagnosis, not for emergencies, not a replacement for a doctor.

## 8. Future work

Real clinical text · more categories · structured inputs (age, duration) · probability calibration · transformer models (BERT / BioBERT / ClinicalBERT) · speech interface · urgency classification (separate project that needs medical validation).

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| Notebook cannot download the data | Download the files manually (section 2) into `data/` and re-run the cell |
| `ValueError: These diseases have no category yet` | Add the listed diseases to `CATEGORY_MAP` in section 5 of the notebook |
| App says "Model file not found" | Run the notebook completely first |
| `InconsistentVersionWarning` when loading the model | Use the same environment (same scikit-learn) for notebook, app and API: `pip install -r requirements.txt` |
| `jupyter` not found | `pip install jupyter`, or open the notebook in VS Code |
| Port already in use | `streamlit run app.py --server.port 8502` or `uvicorn main:app --port 8001` |

---

## 10. Disclaimer

This project is for **educational and research purposes only**. It classifies medical complaints into predefined categories based on patterns learned from a limited, non-clinically-validated dataset. It does not provide medical diagnosis, emergency advice or treatment recommendations. For any health concern, consult a qualified professional; in an emergency, contact your local emergency services.
