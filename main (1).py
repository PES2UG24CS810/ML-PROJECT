# INSINCERE QUESTION CLASSIFICATION (QUORA)

import pandas as pd
import numpy as np
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import joblib
import os

# LOAD DATASET

file_name = None
if os.path.exists("train.csv"):
    file_name = "train.csv"
elif os.path.exists("trainsmall.csv"):
    file_name = "trainsmall.csv"

if file_name:
    print(f" Loading dataset from {file_name}...")
    df = pd.read_csv(file_name)
    df = df.sample(min(1000, len(df)), random_state=42)
else:
    print("Dataset not found — using small sample data instead.")
    data = {
        'question_text': [
            "Why are Muslims so violent?",
            "What is the best way to learn Python?",
            "Is feminism destroying society?",
            "How to cook pasta perfectly?",
            "Are Indians the smartest people?",
            "What causes earthquakes?",
            "Why do politicians lie so much?",
            "How to improve English skills?",
            
            "Why are Americans so fat?",
            "Is there life after death?"
        ],
        'target': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)

print(f" Dataset Loaded: {df.shape[0]} samples")
print(df.head())

# TEXT CLEANING

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()    # lowercasing
    text = re.sub(r"http\S+|www\S+", '', text) 
    text = re.sub(r"[^a-z\s]", '', text)    # tokenization 
    text = " ".join(word for word in text.split() if word not in stop_words)    # stop-word removal
    return text

print("Cleaning text...")
df['clean_text'] = df['question_text'].apply(clean_text)

# TRAIN/TEST SPLIT 

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['target'], test_size=0.2, random_state=42
)

# TF-IDF VECTORIZATION 

print("Applying TF-IDF vectorization...")
vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# MODEL EVALUATION

results = []

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    prec = round(precision_score(y_test, y_pred, zero_division=0), 2)
    rec = round(recall_score(y_test, y_pred, zero_division=0), 2)
    f1 = round(f1_score(y_test, y_pred, zero_division=0), 2)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    results.append([name, f"{acc}%", prec, rec, f1, fp, fn])
    print(f"{name} Done → Accuracy: {acc}%, FP={fp}, FN={fn}")

# Logistic Regression
lr = LogisticRegression(max_iter=300, class_weight='balanced')
evaluate_model("Logistic Regression", lr, X_train_tfidf, y_train, X_test_tfidf, y_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
evaluate_model("Random Forest", rf, X_train_tfidf, y_train, X_test_tfidf, y_test)

# Light Neural Network
mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=20, random_state=42)
evaluate_model("Light Neural Network", mlp, X_train_tfidf, y_train, X_test_tfidf, y_test)

# PRINTING TABLE 

print("\nModel Performance Summary:\n")
print("| Model                | Accuracy   | Precision | Recall | F1-Score | False Positives | False Negatives |")
print("|:---------------------|:-----------|-----------:|-------:|----------:|----------------:|----------------:|")
for row in results:
    model, acc, prec, rec, f1, fp, fn = row
    print(f"| {model:<20} | {acc:<9} | {prec:<9} | {rec:<6} | {f1:<8} | {fp:<16} | {fn:<16} |")

# BEST MODEL 

best_model = results[np.argmax([r[4] for r in results])]  # F1-score
best_model_name = best_model[0]

print(f"\nBest Performing Model: {best_model_name}")
print(f"  F1-Score: {best_model[4]}, FP: {best_model[5]}, FN: {best_model[6]}")

if best_model_name == "Logistic Regression":
    chosen_model = lr
elif best_model_name == "Random Forest":
    chosen_model = rf
else:
    chosen_model = mlp

# VISUALIZATIONS 

# Bar chart for metrics
plt.figure(figsize=(10, 6))
bar_width = 0.2
models = [r[0] for r in results]
x = np.arange(len(models))

plt.bar(x - bar_width, [float(r[1].strip('%')) for r in results], width=bar_width, label="Accuracy")
plt.bar(x, [r[2] for r in results], width=bar_width, label="Precision")
plt.bar(x + bar_width, [r[3] for r in results], width=bar_width, label="Recall")
plt.bar(x + 2*bar_width, [r[4] for r in results], width=bar_width, label="F1-Score")

plt.xticks(x, models, rotation=15)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.show()

# Confusion Matrix for Best Model
y_pred_best = chosen_model.predict(X_test_tfidf)
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Sincere', 'Insincere'],
            yticklabels=['Sincere', 'Insincere'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.show()

# FEATURE IMPORTANCE FOR RANDOM FOREST 

if best_model_name == "Random Forest" or True:
    importances = rf.feature_importances_
    indices = np.argsort(importances)[-20:]  # top 20 features
    top_features = np.array(vectorizer.get_feature_names_out())[indices]
    plt.figure(figsize=(10,6))
    plt.barh(top_features, importances[indices])
    plt.title("Top 20 Feature Importances - Random Forest")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

# SAVE BEST MODEL 

joblib.dump(chosen_model, 'best_quora_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print(f"\nSaved Best Model ({best_model_name}) and Vectorizer Successfully!")
