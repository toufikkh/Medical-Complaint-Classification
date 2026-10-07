# Medical Complaint Classification — Project Specification

## 1. Project Overview

**Project title:** Medical Complaint Classification Using NLP and Machine Learning

The project aims to build a simple Natural Language Processing (NLP) system that takes a patient's textual complaint or symptom description and classifies it into a predefined medical category/specialty.

Example:

```text
"I have red patches on my skin and severe itching."
        ↓
Text preprocessing
        ↓
TF-IDF
        ↓
Machine Learning classifier
        ↓
Dermatology
```

The system is **not a medical diagnosis system**. It does not identify a disease and must not be presented as a clinical decision-making or emergency system.

The correct framing is:

> The system classifies medical complaints into predefined medical categories.

---

## 2. The Problem

Medical clinics and healthcare services can receive many textual descriptions of symptoms and complaints. Manually organizing these complaints can take time.

The proposed system explores whether traditional NLP and machine learning can automatically classify a complaint into the most relevant predefined medical category.

Possible examples:

| Complaint | Possible category |
|---|---|
| Chest pain and shortness of breath | Cardiology |
| Skin rash and itching | Dermatology |
| Persistent cough | Respiratory |
| Tooth pain | Dentistry |
| Stomach pain and nausea | Gastroenterology |

The goal is **classification, not diagnosis**.

---

## 3. Main Research Question

> Can a traditional NLP pipeline using TF-IDF and supervised machine learning accurately classify short medical complaints into predefined medical categories?

---

## 4. Project Scope

The project should remain intentionally simple because the available time is very limited.

### Current MVP

```text
Medical complaint
        ↓
Text cleaning
        ↓
TF-IDF
        ↓
Machine Learning classifier
        ↓
Predicted medical category
        ↓
Streamlit interface
```

The initial implementation should use a classical ML model such as:

- Logistic Regression
- Linear SVM

Start with Logistic Regression. If time permits, compare it with Linear SVM.

Do not add complex models unless there is enough time.

---

## 5. Dataset

The project requires a public dataset containing medical text/symptom descriptions and corresponding labels.

Possible sources include Hugging Face and Kaggle.

Before implementation, inspect the actual dataset and verify:

1. Number of rows
2. Column names
3. Text column
4. Target/label column
5. Number of classes
6. Distribution of classes
7. Missing values
8. Duplicate records
9. Whether the text represents complaints/symptoms
10. Whether labels correspond to meaningful medical categories
11. Whether the dataset is usable for classification

**Do not choose a dataset only because its name sounds suitable.**

---

## 6. Expected Data Structure

Ideally:

```text
complaint/symptoms → category
```

Examples:

```text
"I have severe itching and a red rash" → Dermatology
"I have persistent chest pain" → Cardiology
"I have a severe toothache" → Dentistry
```

The exact columns and labels depend on the selected dataset.

---

## 7. Data Cleaning

Keep preprocessing simple and conservative.

Possible steps:

- Handle missing values
- Remove exact duplicates when appropriate
- Convert text to lowercase
- Remove unnecessary extra whitespace
- Handle obviously irrelevant characters if needed

Do **not** aggressively remove words.

Words such as `no`, `not`, and `without` can be important in medical text.

For example:

```text
"chest pain"
```

is different from:

```text
"no chest pain"
```

---

## 8. Exploratory Data Analysis (EDA)

Keep EDA simple and useful.

Analyze:

- Dataset size
- Number of categories
- Class distribution
- Missing values
- Duplicate records
- Text length distribution if useful

Create a bar chart showing the number of examples per category.

If the classes are imbalanced, mention this in the project.

Do not spend excessive time on EDA.

---

## 9. Text Representation — TF-IDF

Machine learning models cannot directly process raw text.

TF-IDF converts text into numerical features.

**TF-IDF = Term Frequency – Inverse Document Frequency**

Conceptually:

```text
Raw text
   ↓
TF-IDF Vectorizer
   ↓
Numerical feature vectors
```

Words that help distinguish categories can receive higher importance.

---

## 10. Train/Test Split

Use a standard split such as:

```text
80% → Training
20% → Testing
```

Use stratification when appropriate:

```python
train_test_split(..., stratify=y)
```

### Avoid data leakage

The TF-IDF vectorizer must be fitted using the training data only.

```text
Training data
     ↓
TF-IDF fit + transform
     ↓
Model training

Test data
     ↓
TF-IDF transform only
     ↓
Model prediction
```

Do not fit TF-IDF on the complete dataset before splitting.

---

## 11. Machine Learning Model

### Primary model: Logistic Regression

Start with Logistic Regression because it is:

- Simple
- Fast
- Easy to understand
- Suitable for high-dimensional TF-IDF text features
- Appropriate for multiclass classification
- Easy to explain

Pipeline:

```text
Text
 ↓
TF-IDF
 ↓
Logistic Regression
 ↓
Predicted category
```

### Optional second model: Linear SVM

If time permits, train a Linear SVM and compare it with Logistic Regression.

Do not use Random Forest just to have multiple models. Linear models are a natural starting point for sparse TF-IDF text features.

---

## 12. Model Evaluation

Do not rely only on accuracy.

Use:

### Accuracy
Overall percentage of correct predictions.

### Precision
How many predictions for a given class were actually correct.

### Recall
How many examples belonging to a class were correctly identified.

### F1-score
A balance between precision and recall.

### Confusion Matrix
Shows which categories the model confuses.

Example:

```text
                 Predicted
               A    B    C

Actual A      20    2    1
Actual B       3   18    2
Actual C       0    1   25
```

ROC-AUC can be added only if it is appropriate and does not add unnecessary complexity.

---

## 13. Error Analysis

If time permits, inspect a few incorrect predictions.

Example:

```text
Actual: Dermatology
Predicted: Allergy
```

Discuss possible reasons for overlap between categories.

This gives the project a stronger analytical component.

---

## 14. Streamlit Application

Create a simple Streamlit interface.

Example:

```text
---------------------------------------
       Medical Complaint Classifier
---------------------------------------

Enter patient's complaint:

[ I have severe itching and a skin rash ]

             [ Classify ]

---------------------------------------

Predicted Category:
Dermatology

Confidence:
92.4%

---------------------------------------

This application is for educational/research
purposes and does not provide medical diagnosis
or emergency medical advice.
---------------------------------------
```

Application flow:

```text
User enters complaint
        ↓
Text preprocessing
        ↓
Saved TF-IDF vectorizer
        ↓
Saved trained model
        ↓
Prediction
        ↓
Display category
```

---

## 15. Saving the Model

Save:

1. The trained classifier
2. The fitted TF-IDF vectorizer

Use `joblib`.

Example structure:

```text
models/
├── model.pkl
└── tfidf.pkl
```

The Streamlit application loads these files instead of retraining the model.

---

## 16. Suggested Project Structure

```text
medical-complaint-classifier/
│
├── data/
│   └── medical_complaints.csv
│
├── notebooks/
│   └── analysis.ipynb
│
├── models/
│   ├── model.pkl
│   └── tfidf.pkl
│
├── app.py
│
├── requirements.txt
│
└── README.md
```

Keep the structure simple and reduce it further if necessary.

---

## 17. Complete Workflow

```text
Dataset
   ↓
Data inspection
   ↓
Data cleaning
   ↓
EDA
   ↓
Train/Test Split
   ↓
TF-IDF
   ↓
Logistic Regression
   ↓
Evaluation
   ↓
Error analysis (if time permits)
   ↓
Save model + vectorizer
   ↓
Streamlit
```

---

## 18. Presentation Structure

1. **Problem** — Why organizing medical complaints can be useful.
2. **Proposed Solution** — NLP + ML classification.
3. **Dataset** — Source, size, classes, columns, examples.
4. **EDA** — Class distribution and observations.
5. **Preprocessing** — Cleaning and TF-IDF.
6. **Model** — Logistic Regression and optionally Linear SVM.
7. **Evaluation** — Accuracy, Precision, Recall, F1, Confusion Matrix.
8. **Application** — Streamlit demo.
9. **Limitations** — Dataset and model limitations.
10. **Future Work** — Possible extensions.

---

## 19. Medical Safety and Limitations

The system must **not** be presented as:

- A medical diagnosis system
- A doctor replacement
- An emergency triage system
- A system that determines whether someone needs urgent medical attention
- A clinically validated healthcare tool

Use wording such as:

> The model classifies medical complaints into predefined categories based on patterns learned from the training dataset.

And:

> The prototype is intended for educational and research purposes and is not a substitute for professional medical advice.

The quality of the model depends heavily on the quality, representativeness, and labeling of the dataset.

If the dataset is small, synthetic, automatically labeled, or not clinically validated, this must be acknowledged.

---

## 20. Future Development

### Future Version 1 — Additional structured information

Potential features:

- Age
- Temperature
- Symptom duration
- Other structured patient information

### Future Version 2 — Urgency classification

A future system could explore:

```text
Self-monitor
     ↓
Consult doctor
     ↓
Urgent
```

This is **not part of the current MVP**.

### Future Version 3 — Transformer models

Compare traditional NLP with:

- BERT
- BioBERT
- ClinicalBERT

### Future Version 4 — Speech interface

```text
Patient speaks
      ↓
Speech-to-Text
      ↓
Complaint classification
      ↓
Medical category
```

These are future possibilities, not current requirements.

---

## 21. What NOT to Build

Because the available time is very limited, do not add:

- BERT fine-tuning
- RAG
- LLM chatbot
- Medical knowledge base
- Voice recognition
- Complex APIs
- Web scraping
- Complex databases
- Complex authentication
- Medical diagnosis
- Emergency decision-making
- Complex deployment infrastructure

These can be mentioned as future work.

---

## 22. Time Constraint

There are approximately **2 days remaining** to finish the project.

Practical working time:

```text
3 hours/day × 2 days ≈ 6 hours
```

Therefore:

**Speed, simplicity, correctness, and explainability are more important than adding advanced technologies.**

Prioritize a complete working MVP over unnecessary sophistication.

If a component is not necessary for the core objective, skip it.

---

## 23. Core Deliverable

The final result should contain:

```text
Real problem
     +
Real/public dataset
     +
Data cleaning
     +
EDA
     +
NLP preprocessing
     +
TF-IDF
     +
Supervised ML classification
     +
Evaluation
     +
Simple Streamlit application
```

The project should be understandable and defensible by a beginner Data Science student.

---

## 24. Main Development Principle

> Keep the project simple, real, measurable, and explainable.

Do not add technology just to make the project look more advanced.

A clean and well-evaluated TF-IDF + Logistic Regression project is preferable to a complicated system that cannot be completed or explained.

### Immediate Next Step

**Find and inspect a suitable real dataset before implementing the model.**
