import joblib
import numpy as np
from scipy.sparse import hstack

model = joblib.load("xgb_depression_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

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
        tfidf_features = vectorizer.transform([sentence])
        print(f"TF-IDF Features Shape: {tfidf_features.shape}")

        num_characters = len(sentence)
        num_sentences = sentence.count('.') + sentence.count('!') + sentence.count('?')
        num_features = np.array([[num_characters, num_sentences]])
        print(f"Numerical Features: {num_features}")

        combined_features = hstack([tfidf_features, num_features])
        print(f"Combined Features Shape: {combined_features.shape}")

        probabilities = model.predict_proba(combined_features)
        print(f"Prediction Probabilities: {probabilities}")

        predicted_label = np.argmax(probabilities, axis=1)[0]

        return labels[predicted_label]
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {e}"

sample_text = "She comes crying to me and formulates a plan to break up. She talks to Joe about their issues and her will to leave him wilts. She stays with him. Rinse and repeat, except it gets worse over time. How can I break the cycle, or help her break the cycle?"
result = detect_depression(sample_text)
print(f"Prediction: {result}")