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
    f1_score, confusion_matrix, classification_report
)
import joblib
import os
import time

# DATA LOADING

file_name = "train.csv"

if os.path.exists(file_name):
    print(f"Loading dataset from {file_name}...")
    df = pd.read_csv(file_name, on_bad_lines='skip')
    df = df.sample(min(10000, len(df)), random_state=42)
    print(f"Dataset Loaded: {df.shape[0]} samples")
else:
    print("Using sample dataset instead.")
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
    print(f"Sample Dataset Loaded: {df.shape[0]} samples")

# TEXT CLEANING

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r"[^a-z\s]", '', text)
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

print("Cleaning text data...")
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

# MODEL TRAINING AND EVALUATION

results = []

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print(f"\nTraining {name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - start_time, 2)
    y_pred = model.predict(X_test)
    
    acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    prec = round(precision_score(y_test, y_pred, zero_division=0), 2)
    rec = round(recall_score(y_test, y_pred, zero_division=0), 2)
    f1 = round(f1_score(y_test, y_pred, zero_division=0), 2)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    results.append([name, f"{acc}%", prec, rec, f1, fp, fn, train_time])
    print(f"{name} done → Accuracy: {acc}%, Time: {train_time}s")

# Initialize models
lr = LogisticRegression(max_iter=300, class_weight='balanced')
rf = RandomForestClassifier(n_estimators=100, random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=20, random_state=42)

# Evaluate models
evaluate_model("Logistic Regression", lr, X_train_tfidf, y_train, X_test_tfidf, y_test)
evaluate_model("Random Forest", rf, X_train_tfidf, y_train, X_test_tfidf, y_test)
evaluate_model("Light Neural Network", mlp, X_train_tfidf, y_train, X_test_tfidf, y_test)

# PERFORMANCE SUMMARY TABLE

print("\nModel Performance Summary:\n")
print("| Model                | Accuracy | Precision | Recall | F1 | FP | FN | Train Time (s) |")
print("|:---------------------|:---------|:----------|:------:|:--:|:--:|:--:|:--------------:|")
for row in results:
    print(f"| {row[0]:<20} | {row[1]:<8} | {row[2]:<9} | {row[3]:<6} | {row[4]:<4} | {row[5]:<2} | {row[6]:<2} | {row[7]:<14} |")

# PERFORMANCE VISUALIZATION

plt.figure(figsize=(10, 6))
bar_width = 0.2
models_list = [r[0] for r in results]
x = np.arange(len(models_list))

plt.bar(x - bar_width, [float(r[1].strip('%')) for r in results], width=bar_width, label="Accuracy")
plt.bar(x, [r[2] for r in results], width=bar_width, label="Precision")
plt.bar(x + bar_width, [r[3] for r in results], width=bar_width, label="Recall")
plt.bar(x + 2*bar_width, [r[4] for r in results], width=bar_width, label="F1-Score")

plt.xticks(x, models_list, rotation=15)
plt.ylabel("Metric Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.show()

# BEST MODEL SELECTION

best_model = results[np.argmax([r[4] for r in results])]  # highest F1-score
best_model_name = best_model[0]
chosen_model = {'Logistic Regression': lr, 'Random Forest': rf, 'Light Neural Network': mlp}[best_model_name]

print(f"\nBest Performing Model: {best_model_name} with F1-Score = {best_model[4]}")

# CONFUSION MATRIX

y_pred_best = chosen_model.predict(X_test_tfidf)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Sincere', 'Insincere'],
            yticklabels=['Sincere', 'Insincere'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix – {best_model_name}')
plt.tight_layout()
plt.show()

# FEATURE IMPORTANCE

feature_names = np.array(vectorizer.get_feature_names_out())

# Logistic Regression – top coefficients
coefs = lr.coef_[0]
top_positive_indices = np.argsort(coefs)[-10:]
top_negative_indices = np.argsort(coefs)[:10]
plt.figure(figsize=(10, 6))
plt.barh(feature_names[top_negative_indices], coefs[top_negative_indices], color="red", label="Sincere")
plt.barh(feature_names[top_positive_indices], coefs[top_positive_indices], color="green", label="Insincere")
plt.title("Top Features – Logistic Regression")
plt.xlabel("Coefficient Weight")
plt.ylabel("Feature")
plt.legend()
plt.tight_layout()
plt.show()

# Random Forest – top feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[-15:]
plt.figure(figsize=(10, 6))
plt.barh(feature_names[indices], importances[indices], color='teal')
plt.title("Top 15 Important Features – Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# Light Neural Network – approximate input importance
input_weights = mlp.coefs_[0]
mean_abs_weights = np.mean(np.abs(input_weights), axis=1)
indices = np.argsort(mean_abs_weights)[-15:]
plt.figure(figsize=(10, 6))
plt.barh(feature_names[indices], mean_abs_weights[indices], color='purple')
plt.title("Top 15 Important Features – Light Neural Network")
plt.xlabel("Average Absolute Weight")
plt.tight_layout()
plt.show()

# CLASSIFICATION REPORT

print("\nDetailed Classification Report:\n")
print(classification_report(y_test, y_pred_best, target_names=['Sincere', 'Insincere']))

# SAVE MODEL

joblib.dump(chosen_model, 'best_quora_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print(f"\nModel and vectorizer saved successfully → {best_model_name}")
print("\nTraining, Evaluation, and Visualization Completed Successfully!")
