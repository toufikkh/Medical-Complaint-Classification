"""
Medical Complaint Classifier - Streamlit app
============================================
Run with:   streamlit run app.py

What it does
------------
1. Loads the trained pipeline (TF-IDF + Logistic Regression) from models/pipeline.joblib
   (created by running notebooks/analysis.ipynb).
2. Takes a free-text complaint typed by the user.
3. Returns the TOP-3 most likely medical CATEGORIES with a probability for each.

This is NOT a diagnosis tool. It classifies text into predefined categories only.
"""

import json
import re
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "pipeline.joblib"
META_PATH = BASE_DIR / "models" / "metadata.json"

TOP_K = 3                 # how many categories to display
LOW_CONFIDENCE = 0.50     # below this top probability we show a warning
MAX_CHARS = 2000          # protect the app from huge inputs

DISCLAIMER = (
    "This application is for educational/research purposes only. It classifies a text "
    "complaint into a predefined medical category based on patterns learned from a small "
    "dataset. It does NOT provide a medical diagnosis, is NOT an emergency or triage tool, "
    "and is not a substitute for professional medical advice. If you are in an emergency, "
    "contact your local emergency services."
)

EXAMPLES = [
    "I have red patches on my skin and severe itching.",
    "I have a persistent cough and difficulty breathing, especially at night.",
    "I feel a burning pain in my chest after eating and I often have acid in my mouth.",
    "I have a pounding headache on one side with nausea and sensitivity to light.",
    "I have joint pain and stiffness in my knees, worse in the morning.",
]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Same cleaning as in the notebook (keep identical!).

    Conservative on purpose: lowercase, remove odd characters, collapse spaces.
    Stop-words are NOT removed, so negations like 'no', 'not', 'without' are kept.
    """
    text = str(text).lower()
    text = re.sub(r"[^\w\s.,'\-/%]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the saved pipeline once and keep it in memory."""
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    return {}


def predict_top_k(model, text: str, k: int = TOP_K):
    """Return a list of (category, probability) sorted from most to least likely."""
    proba = model.predict_proba([clean_text(text)])[0]
    top_idx = np.argsort(proba)[::-1][:k]
    return [(str(model.classes_[i]), float(proba[i])) for i in top_idx]


def set_example(example_text: str):
    """Callback used by the example buttons."""
    st.session_state["complaint"] = example_text


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Medical Complaint Classifier", page_icon="🩺", layout="centered")
    st.title("🩺 Medical Complaint Classifier")
    st.caption("NLP project: TF-IDF + Logistic Regression -> top-3 medical categories")
    st.info(DISCLAIMER)

    # --- Model availability check -------------------------------------------
    if not MODEL_PATH.exists():
        st.error(
            "Model file not found: `models/pipeline.joblib`.\n\n"
            "Run **notebooks/analysis.ipynb** from top to bottom first. "
            "It trains the model and saves it into the `models/` folder."
        )
        st.stop()

    model = load_model()
    meta = load_metadata()

    # --- Sidebar: model information -----------------------------------------
    with st.sidebar:
        st.header("About the model")
        st.write("**Categories the model knows:**")
        for c in model.classes_:
            st.write(f"- {c}")
        metrics = meta.get("metrics", {})
        if metrics:
            st.write("**Test-set results** (from the notebook):")
            st.write(f"- Accuracy: {metrics.get('accuracy', float('nan')):.1%}")
            st.write(f"- Macro F1: {metrics.get('macro_f1', float('nan')):.1%}")
            st.write(f"- Top-3 accuracy: {metrics.get('top3_accuracy', float('nan')):.1%}")
        st.caption(
            "The model only knows the categories listed above. A complaint from an unknown "
            "specialty (for example dental) will still be forced into one of them."
        )

    # --- Input ---------------------------------------------------------------
    st.subheader("Enter a patient's complaint")
    st.text_area(
        "Describe the symptoms in your own words:",
        key="complaint",
        height=140,
        max_chars=MAX_CHARS,
        placeholder="e.g. I have severe itching and a red rash on my arms",
    )

    st.write("Or try an example:")
    for i, ex in enumerate(EXAMPLES):
        st.button(ex, key=f"ex_{i}", on_click=set_example, args=(ex,), use_container_width=True)

    # --- Prediction -----------------------------------------------------------
    if st.button("Classify", type="primary"):
        complaint = st.session_state.get("complaint", "")
        if not clean_text(complaint):
            st.warning("Please type a complaint first.")
        else:
            results = predict_top_k(model, complaint, TOP_K)

            st.subheader("Most likely categories")
            for rank, (category, prob) in enumerate(results, start=1):
                left, right = st.columns([3, 1])
                left.markdown(f"**{rank}. {category}**")
                right.markdown(f"**{prob:.1%}**")
                st.progress(min(max(prob, 0.0), 1.0))

            if results[0][1] < LOW_CONFIDENCE:
                st.warning(
                    "Low confidence: the model is not sure about this complaint. "
                    "The text may be vague, or may not belong to the categories it knows."
                )
            st.caption(
                "Percentages are model scores learned from a small dataset, "
                "not real medical probabilities."
            )

    st.divider()
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
