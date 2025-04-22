import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit App
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")
st.write("Enter a news article or statement below to check whether it's fake or real.")

# Input box
user_input = st.text_area("News Content", height=200)

# Predict button
if st.button("Detect"):
    if user_input.strip():
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        
        if prediction == 1:
            st.success("✅ The news appears to be **REAL**.")
        else:
            st.error("🚫 The news appears to be **FAKE**.")
    else:
        st.warning("Please enter some news content to analyze.")
