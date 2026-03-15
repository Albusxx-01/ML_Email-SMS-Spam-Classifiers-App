import streamlit as st
import pickle
import string
import nltk
nltk.download()
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    text = y[:]
    y = [ps.stem(i) for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    return " ".join(y)


# Load the pre-trained models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# --- Page Configuration ---
st.set_page_config(page_title="Spam Detector", page_icon="🚫", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header Section ---
st.title("🛡️ Email/SMS Spam Classifier")
st.markdown("Analyze your messages instantly to identify potential spam or phishing attempts.")
st.divider()

# --- Input Section ---
input_sms = st.text_area("Paste the message you want to scan below:", height=150,
                         placeholder="e.g., CONGRATULATIONS! You've won a $500 gift card...")

if st.button("Analyze Message"):

    if input_sms.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner('Analyzing patterns...'):
            # 1. Preprocess
            transformed_sms = transform_text(input_sms)
            # 2. Vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3. Predict
            result = model.predict(vector_input)[0]
            # 4. Probabilities (Optional: shows how sure the model is)
            proba = model.predict_proba(vector_input)[0]

        st.divider()

        # --- Display Results ---
        col1, col2 = st.columns([1, 1])

        with col1:
            if result == 1:
                st.error("### 🚨 Result: SPAM")
            else:
                st.success("### ✅ Result: NOT SPAM")

        with col2:
            confidence = proba[1] if result == 1 else proba[0]
            st.metric("Confidence Score", f"{confidence * 100:.2f}%")

        if result == 1:
            st.info("**Safety Tip:** Do not click any links or provide personal information to this sender.")

# --- Footer ---
st.markdown("---")
st.caption("Developed by Satwik | Powered by MultinomialNB & Scikit-Learn")