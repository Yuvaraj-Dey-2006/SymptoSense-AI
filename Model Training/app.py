"""
app.py
Streamlit interface for the AI Medical Symptom Assistant.

Run: streamlit run app.py
Requires: streamlit, scikit-learn, joblib
(model artifacts must already exist - run train_model.py first)
"""

import random
import joblib
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="AI Medical Symptom Assistant", page_icon="🩺")
st.title("🩺 AI Medical Symptom Assistant")
st.caption(
    "This assistant covers cancer, diabetes, Alzheimer's, hypertension, and stroke. "
    "It is not a substitute for professional medical advice."
)

# -----------------------------
# Load model artifacts (cached so it only loads once per session)
# -----------------------------
@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load("vectorizer.pkl")
    clf = joblib.load("classifier.pkl")
    tag_to_responses = joblib.load("tag_to_responses.pkl")
    return vectorizer, clf, tag_to_responses

vectorizer, clf, tag_to_responses = load_artifacts()

CONFIDENCE_THRESHOLD = 0.15

def get_response(user_input: str) -> str:
    X_input = vectorizer.transform([user_input])
    probs = clf.predict_proba(X_input)[0]
    max_prob = probs.max()
    predicted_tag = clf.classes_[probs.argmax()]

    if max_prob < CONFIDENCE_THRESHOLD:
        return "I'm not sure I understand. Could you rephrase that?"

    responses = tag_to_responses.get(predicted_tag, ["I don't have information on that."])
    return random.choice(responses)

# -----------------------------
# Chat state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I assist you with Medical Assistance today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# -----------------------------
# Chat input
# -----------------------------
if user_input := st.chat_input("Ask about symptoms, prevention, or types of a condition..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    response = get_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)
