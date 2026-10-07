# Explanations, Defense Questions and Presentation Outline

This document helps you **understand and defend** every part of the project.

---

## Part A: Concepts in plain language

### A1. What problem are we solving?
Clinics receive text descriptions of symptoms. Sorting them by hand takes time. We test whether a classical NLP + ML pipeline can **sort a complaint into a medical category** automatically. The output is a **ranked list of categories with probabilities**, not a diagnosis.

### A2. Why categories and not diseases?
- Fewer classes (about 10 instead of 20+) means more examples per class and a more reliable model.
- Disease prediction is medically sensitive and needs far more, better data.
- Category sorting is realistic for a small project and is what the project specification asks for.

### A3. Text cleaning
We lowercase, remove odd characters and collapse whitespace. We **do not remove stop-words**, because "no", "not" and "without" change meaning: "no chest pain" is the opposite of "chest pain". We also keep word pairs (bigrams) in TF-IDF so that phrases like "no cough" can be a feature.

### A4. TF-IDF
Computers need numbers, not words. TF-IDF gives every word a weight in every text:

- **TF (term frequency):** how often the word appears in this text.
- **IDF (inverse document frequency):** how rare the word is across all texts. Rare words get a higher IDF.
- Weight = TF x IDF (scikit-learn also normalises each text vector).

So "itching" (rare, specific) gets more weight than "have" (everywhere). Result: each complaint becomes a long vector of numbers (one per vocabulary word or word pair), mostly zeros. This is called a *sparse* vector.

### A5. Logistic Regression for many categories
For each category the model learns a weight for every word. For a new text it adds up the weights and converts the scores to **probabilities that sum to 100%** (softmax). That is why we can display "Dermatology 71%, Allergy 15%, ...". Reasons to choose it: simple, fast, works very well on TF-IDF features, outputs probabilities, and is **explainable** (we can list the strongest words per category).

### A6. Why `class_weight="balanced"`?
Some categories have more examples than others (e.g. Infectious Disease has 4 diseases, Neurology has 1). Without balancing, the model favours big categories. Balancing gives small categories more weight during training.

### A7. Train/test split and stratification
We hide 20% of the data (the test set) and never touch it until the end. **Stratified** means each category keeps the same proportion in train and test. Test results estimate how the model behaves on new data.

### A8. Data leakage
Leakage = information from the test set sneaking into training, which makes results look better than they are. Typical mistake: fitting TF-IDF on the whole dataset before splitting. We prevent this with a `Pipeline`, and by dropping duplicate texts before splitting (otherwise the same text could be in both sets).

### A9. Cross-validation and tuning
We try a few settings (`C`, with or without word pairs). 5-fold cross-validation splits the **training set** into 5 parts, trains on 4, checks on 1, and rotates. We choose the setting with the best macro-F1. The test set stays untouched.

### A10. Metrics
- **Accuracy:** correct predictions / all predictions. Can hide weak categories.
- **Precision (per category):** of the texts predicted as X, how many really are X.
- **Recall (per category):** of the texts that really are X, how many we found.
- **F1:** the balance of precision and recall.
- **Macro-F1:** average F1 over categories; every category counts equally.
- **Confusion matrix:** table of actual vs predicted; shows which categories get mixed up.
- **Top-3 accuracy:** the true category is among the 3 shown. Matches the app.
- **Baseline:** "always predict the most common category". Our model must beat it.

### A11. Why is the percentage not a real probability?
The numbers are the model's confidence given what it learned from a small dataset. They are **not calibrated** (70% does not mean 70 out of 100 such patients have that condition). We say this in the app.

### A12. Why does the model never say "I don't know"?
It is a closed-set classifier: it must spread 100% over the categories it knows. A dental complaint will still get categories like "Infectious Disease". The only protection we added is a **low-confidence warning** when the top probability is below 50%.

---

## Part B: Questions you may be asked (with short answers)

**1. Why TF-IDF and not BERT?**
Project scope and time. TF-IDF + a linear model is fast, explainable, and strong for short keyword-driven texts. BERT/BioBERT are listed as future work.

**2. Why Logistic Regression?**
Linear models suit sparse high-dimensional TF-IDF features, train in seconds, give probabilities, and are easy to explain.

**3. Why compare with a Linear SVM, and why do you keep Logistic Regression?**
SVM is another natural linear model for text. It has no direct probabilities, and the app needs percentages, so Logistic Regression is used for deployment.

**4. Why not Random Forest or deep learning?**
They are not a natural fit for sparse text features and would add complexity without a clear reason. We do not add models just to look advanced.

**5. Your accuracy is very high. Is the model really that good?**
Not necessarily. The dataset is small, clean and probably synthetic, with consistent wording per class. Real patient text would give lower performance. We report this as a limitation instead of claiming real-world accuracy.

**6. How did you avoid data leakage?**
Duplicates removed before splitting; stratified split with a fixed seed; TF-IDF inside a `Pipeline` so it is fitted on training data only; tuning by cross-validation on the training set only.

**7. Why didn't you remove stop-words?**
Negations ("no", "not", "without") carry medical meaning.

**8. How did you create the categories?**
With one explicit disease-to-category dictionary. It is a manual judgment, not clinically validated, and some diseases could belong to several categories. This is a documented limitation.

**9. What does the top-3 output add?**
Complaints often fit more than one category. Showing several with probabilities is more honest than one hard answer, and top-3 accuracy measures this.

**10. What happens with a complaint outside the known categories?**
It is still assigned to some category, usually with low confidence, which triggers the warning. A future version could add an "unknown" class or a rejection threshold.

**11. Is this a diagnosis tool?**
No. It sorts text into categories, is not clinically validated, and must not be used for emergencies or decisions about care.

**12. What would you improve with more time?**
Real, larger and clinically validated data; probability calibration; transformer models; an "unknown" option; evaluation by medical experts.

**13. Why one `.joblib` file instead of separate model and vectorizer?**
A `Pipeline` bundles both, guaranteeing that the vectorizer matches the model and the same steps are applied at prediction time.

**14. What do the confusion matrix and error analysis tell you?**
Which categories are mixed up (often categories with shared symptoms) and whether errors come from vague text, overlap between categories, or our own mapping.

---

## Part C: Presentation outline (about 10 minutes, 10 slides)

| # | Slide | What to say / show |
|---|---|---|
| 1 | Title | Project name, your name, course. One line: "classification, not diagnosis" |
| 2 | Problem | Clinics receive many text complaints; sorting by hand is slow |
| 3 | Solution | Diagram: complaint -> cleaning -> TF-IDF -> Logistic Regression -> top-3 categories |
| 4 | Dataset | Source and link, number of rows, texts and labels (2 examples), the disease-to-category mapping and why |
| 5 | EDA | `reports/class_distribution.png`; mention imbalance and duplicates removed |
| 6 | Preprocessing | Cleaning choices; why stop-words are kept; TF-IDF in one sentence |
| 7 | Model | Pipeline, tuning, baseline, optional SVM comparison |
| 8 | Evaluation | Metrics table (your numbers), `reports/confusion_matrix.png`, 2 error examples |
| 9 | Demo | Live Streamlit app: 2 good examples, 1 negation, 1 out-of-scope (dental) showing low confidence. Optionally show `/docs` of the API |
| 10 | Limitations and future work | Small/synthetic data, manual mapping, closed categories, not clinical. Future: real data, calibration, BERT, "unknown" class |

**Demo tips**
- Run the app before the presentation and keep one browser tab ready.
- Prepare 4 sentences in advance (see the examples at the end of the notebook).
- Say clearly at the start of the demo: "This is not a diagnosis".

---

## Part D: Pre-submission checklist

- [ ] Notebook runs top-to-bottom (`Restart & Run All`) without errors
- [ ] "Inspection conclusions" and "Error analysis notes" cells filled in
- [ ] Metrics table in README filled with your real numbers
- [ ] Figures present in `reports/`
- [ ] App starts and shows top-3 with percentages
- [ ] API `/docs` works and `/predict` returns results
- [ ] Disclaimer visible in the app, README and slides
- [ ] Limitations slide mentions small/synthetic data and the manual mapping
