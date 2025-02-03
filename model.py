import joblib
import numpy as np
from scipy.sparse import hstack

# Load the saved model and vectorizer
model = joblib.load("xgb_depression_model.joblib")  # Ensure the correct file path
vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Ensure the correct file path

# Define label mapping
labels = [
    "Anxiety",
    "Bipolar",
    "Depression",
    "Normal",
    "Personality disorder",
    "Stress",
    "Suicidal"
]

def detect_depression(sentence):
    try:
        # Step 1: Transform the input sentence into TF-IDF features
        tfidf_features = vectorizer.transform([sentence])
        print(f"TF-IDF Features Shape: {tfidf_features.shape}")  # Debugging

        # Step 2: Extract numerical features from the input sentence
        num_characters = len(sentence)
        num_sentences = sentence.count('.') + sentence.count('!') + sentence.count('?')
        num_features = np.array([[num_characters, num_sentences]])
        print(f"Numerical Features: {num_features}")  # Debugging

        # Step 3: Combine TF-IDF features with numerical features
        combined_features = hstack([tfidf_features, num_features])
        print(f"Combined Features Shape: {combined_features.shape}")  # Debugging

        # Step 4: Get prediction probabilities (for multi-class models)
        probabilities = model.predict_proba(combined_features)
        print(f"Prediction Probabilities: {probabilities}")  # Debugging

        # Step 5: Choose the label with the highest probability
        predicted_label = np.argmax(probabilities, axis=1)[0]

        return labels[predicted_label]  # Map prediction to corresponding label
    except Exception as e:
        print(f"Error: {e}")  # Debugging
        return f"Error: {e}"