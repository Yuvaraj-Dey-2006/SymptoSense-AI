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
st.set_page_config(page_title="SymptoSense AI | AI Medical Symptom Assistant", page_icon="🩺")
st.title("🩺 SymptoSense AI")
st.caption(
    "🩺 Welcome to SymptomSense AI!\n\n"
    "Not feeling your best? Describe your symptoms in your own words, and I'll help identify the most relevant medical category and provide helpful information.\n\n"
    "Supporting insights for cancer, diabetes, Alzheimer's disease, hypertension, and stroke.\n\n"
    "⚠️ Remember: I'm here to inform, not diagnose. Always consult a healthcare professional for medical advice."
)

st.markdown("""
<style>

/* Entire chat area */
.chat-container{
    display:flex;
    flex-direction:column;
    gap:18px;
    margin-top:25px;
}

/* Assistant */
.bot-row{
    display:flex;
    justify-content:flex-start;
    margin: 14px 0;
}

.bot-bubble{
    background:transparent;
    color:white;
    max-width:85%;
    line-height:1.7;
    font-size:17px;
}

/* User */
.user-row{
    display:flex;
    justify-content:flex-end;
    margin: 14px 0;
}

.chat-container{
    display:flex;
    flex-direction:column;
    gap:24px;
}

.user-bubble{
    background:#2b2b2b;
    color:white;
    padding:12px 18px;
    border-radius:22px;
    display:inline-block;
    width:fit-content;
    max-width:45%;
    word-wrap:break-word;
    font-size:16px;
}

/* Small avatar */
.avatar{
    font-size:24px;
    margin-right:10px;
}

.bot-wrapper{
    display:flex;
    align-items:flex-start;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load model artifacts (cached so it only loads once per session)
# -----------------------------
@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load(r"..\Model\vectorizer.pkl")
    clf = joblib.load(r"..\Model\classifier.pkl")
    tag_to_responses = joblib.load(r"..\Model\tag_to_responses.pkl")
    return vectorizer, clf, tag_to_responses

vectorizer, clf, tag_to_responses = load_artifacts()

print("=" * 50)
print("Vectorizer:", type(vectorizer), vectorizer)
print("Classifier:", type(clf), clf)

print("Has transform:", hasattr(vectorizer, "transform"))
print("Has predict_proba:", hasattr(clf, "predict_proba"))
print("Has multi_class:", hasattr(clf, "multi_class"))
print("=" * 50)

CONFIDENCE_THRESHOLD = 0.25

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
        {
            "role": "assistant",
            "content": "Hello! How can I assist you with Medical Assistance today?"
        }
    ]

# -----------------------------
# Render chat
# -----------------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for message in st.session_state.messages:

    if message["role"] == "assistant":

        st.markdown(f"""
        <div class="bot-row">
            <div class="bot-wrapper">
                <span class="avatar">🤖</span>
                <div class="bot-bubble">
                    {message["content"]}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="user-row">
            <div class="user-bubble">
                {message["content"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Chat input
# -----------------------------
if user_input := st.chat_input(
    "Ask about symptoms, prevention, or types of a condition..."
):

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    response = get_response(user_input)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    st.rerun()