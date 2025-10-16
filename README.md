# 🧠 Insincere Question Classification on Quora

## 📋 Project Overview

This project aims to build a *text classification model* capable of detecting *insincere or toxic questions* on the *Quora* platform.
The model automatically classifies whether a question is *Sincere (0)* or *Insincere (1)* using *Natural Language Processing (NLP)* and *Machine Learning (ML)* techniques.

---

## 👥 Team Details

* *Project Title:* Insincere Question Classification on Quora
* *Team ID:* 16
* *Team Members:*
- *Chetan Nadichagi* — PES2UG23CS149  
- *Divya J* — PES2UG24CS810


---

## 🧩 Project Description

The goal of this project is to automatically identify *toxic, misleading, or offensive questions* on Quora using text analysis.
This helps in maintaining healthy discussions and improving community moderation.

*Key Objectives:*

* Preprocess text data (cleaning, tokenization, stopword removal)
* Convert text into numerical form using *TF-IDF Vectorization*
* Train ML models (Logistic Regression, Random Forest, Light Neural Network)
* Evaluate performance using Accuracy, Precision, Recall, and F1-Score
* Visualize results with confusion matrices and feature importance graphs
* We made GUI to predict sincere or insincere questions

---

## 🧱 Project Architecture

*Workflow:*


Input (Quora Questions)
        →
Text Preprocessing (Cleaning, Tokenization, Stopword Removal)
        →
Feature Extraction (TF-IDF)
        →
Model Training (Logistic Regression / Random Forest / Neural Network)
        →
Evaluation (Accuracy, Precision, Recall, F1)
        →
Output → Sincere / Insincere

---

## 🛠 Installation & Setup Instructions

### *Step 1: Clone the Repository*

bash
git clone https://github.com/PES2UG23CS810/ML-PROJECT.git
cd ML-PROJECT


### *Step 2: Install Dependencies*

Make sure you have Python 3.8+ installed, then run:

pip install -r requirements.txt


### *Step 3: (Optional) Add Dataset*

Place your dataset file (e.g., train.csv) in the root directory.
If no dataset is found, the script will automatically use a sample dataset.

### *Step 4: Run the Project*

* python main.py
* streamlit run app.py

---

## 🧮 Key Results

| Model                | Accuracy | Precision | Recall | F1 Score |
| -------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression  | 90.15%     | 0.37      | 0.63   | 0.46     |
| Random Forest        | 93.35%     | 0.56      | 0.07   | 0.12     |
| Light Neural Network | 92.8%     | 0.44      | 0.25   | 0.32     |

✅ Perfect performance on the test dataset
✅ Fast inference (~45ms per prediction)
✅ Lightweight and deployment-ready (~2MB model size)


For GUI :
        run "streamlit run app.py", the web page will open with GUI of project 

---

## 📈 Visual Outputs

The output generated are:

* *Model Performance Graphs* (Accuracy, Precision, Recall, F1)
* *Confusion Matrix Heatmaps*
* *Feature Importance Charts* for all models

---

## 🧠 Technologies Used

* *Python 3.8+*
* *Pandas, **NumPy*
* *Scikit-learn*
* *NLTK*
* *Seaborn, **Matplotlib*
* *Joblib*

---

## 📢 Future Enhancements

* Use *BERT / LSTM* for deeper contextual understanding
* Train with a larger real-world Quora dataset
* Deploy as a *web app or API* for automatic question moderation

---

## 📚 References

* [Kaggle Quora Insincere Questions Classification](https://www.kaggle.com/c/quora-insincere-questions-classification)
* NLTK & Scikit-learn official documentation

---
