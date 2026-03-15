# Text Classification Web App (Streamlit)

## Introduction
This project is a deployed web application built using Streamlit that allows users to classify text using a trained machine learning model. The application takes user input text, processes it using natural language preprocessing techniques, and predicts its category using a trained TF-IDF + Multinomial Naive Bayes model.

The app is deployed on Streamlit Cloud and provides an interactive interface for real-time text classification.

---

## Live Application
You can try the deployed app here:

Streamlit App Link: https://sms-spam-classifier-mnb.streamlit.app/

---

## Features
- Simple and interactive web interface
- Real-time text prediction
- Automatic text preprocessing
- Uses trained machine learning model
- Deployed on Streamlit Cloud

---

## How the App Works

1. User enters a text message in the input box.
2. The text goes through preprocessing steps:
   - Lowercase conversion
   - Tokenization
   - Removal of special characters
   - Stopword removal
   - Stemming
3. The cleaned text is converted into numerical form using TF-IDF Vectorizer.
4. The trained Multinomial Naive Bayes model predicts the class of the text.
5. The prediction result is displayed instantly on the interface.

---

## Machine Learning Model

Vectorization Technique  
TF-IDF Vectorizer

Classification Model  
Multinomial Naive Bayes

This combination was selected after comparing multiple models and choosing the one with the best precision and overall performance.

---

## Technologies Used

Python  
Streamlit  
Scikit-learn  
Pandas  
NumPy  
NLTK  

---

## Project Structure

Text-Classification-App/

app.py  
model.pkl  
vectorizer.pkl  
requirements.txt  
README.md  

---

## Installation and Running Locally

Clone the repository:

Install dependencies:

Run the Streamlit app:


The application will open in your browser.

---

## Deployment

The application is deployed using Streamlit Cloud. The deployment process includes:

1. Uploading the project to GitHub
2. Connecting the repository to Streamlit Cloud
3. Deploying the app directly from the repository

---

## Conclusion

This project demonstrates how a trained machine learning model can be integrated into a web application using Streamlit. It provides an end-to-end workflow from model training to deployment, making machine learning models accessible through an interactive interface.
