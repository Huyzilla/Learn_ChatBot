import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from data_intent import training_data

# Chuyen du lieu thanh DataFrame
rows = [] 

for intent, examples in training_data.items():
    for example in examples:
        rows.append({'text': example, 'intent': intent})

df = pd.DataFrame(rows)

# Xay dung pipeline
# 1. TFidfVectorizer: Chuyen van ban thanh vector 
# 2. LogisticRegression: Mo hinh phan loai
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()), # Buoc 1 
    ('clf', LogisticRegression(random_state=42)), # Buoc 2
])

# Train 
print("Training model...")
text_clf.fit(df['text'], df['intent'])
print("Model trained successfully.")

# Luu mo hinh
joblib.dump(text_clf, 'intent_classifier.pkl')
print("Model saved to 'intent_classifier.pkl'.")


