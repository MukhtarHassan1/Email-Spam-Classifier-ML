# Email-Spam-Classifier-ML
AI-powered spam email classifier using NLP and Machine Learning (Naive Bayes, TF-IDF)

# 📧 Email Spam Classifier – NLP & Machine Learning Project

An AI-powered text classification system that detects and labels incoming messages (emails or SMS) as **"Spam"** or **"Not Spam"** using **Natural Language Processing (NLP)** and **Machine Learning (ML)**. This project demonstrates how real-world email filtering systems work using Python, NLP pipelines, and a trained ML model.


## Problem Statement

Spam emails are not only annoying but can be harmful, containing phishing links or scams. Manual filtering is inefficient. The goal of this project is to automate spam detection using machine learning, making message classification faster, smarter, and more accurate.

## Objectives

- Build a complete ML pipeline to classify messages.
- Preprocess and clean the textual data using NLP.
- Extract meaningful features using TF-IDF.
- Train a classification model to detect spam messages.
- Save the trained model and vectorizer for future predictions.


## Features

- Classification of text into **Spam** or **Not Spam**
- Real-time predictions using a saved model
- Text preprocessing pipeline:  
  - Lowercasing  
  - Tokenization  
  - Stopword removal  
  - Stemming (via NLTK)
- Feature extraction using **TF-IDF**
- Trained using **Multinomial Naive Bayes** – efficient for text data
- Model & vectorizer saved using **Pickle**
- Built and tested in a **Jupyter Notebook**

---

## Tools & Technologies Used

- **Language**: Python 3.x  
- **Libraries used are**:  
  - `Scikit-learn`  
  - `NLTK`  
  - `Pandas`, `NumPy`  
  - `Pickle`  
  - `Jupyter Notebook`

---

## Dataset

- **Source**: Public dataset from Kaggle
- Contains thousands of labeled messages as “ham” (not spam) or “spam”
- Balanced dataset suitable for binary classification

---

## Project Structure
email-spam-classifier-ml/
├── data/
│   └── spam.csv
│       → The dataset used for training and testing. It contains labeled text messages as 'spam' or 'ham' (not spam).
│
├── model/
│   ├── spam_model.pkl
│       → The trained Multinomial Naive Bayes model saved using Pickle. Used for making real-time predictions.
│   └── vectorizer.pkl
│       → The TF-IDF vectorizer that transforms text into numerical format. Also saved using Pickle.
│
├── spam_classifier.ipynb
│   → A complete Jupyter Notebook containing:
│      - Data cleaning and preprocessing
│      - Feature extraction (TF-IDF)
│      - Model training and evaluation
│      - Saving the model and vectorizer
│
├── requirements.txt
│   → A list of Python packages needed to run this project. Install with `pip install -r requirements.txt`.
│
└── README.md
    → Project documentation explaining purpose, features, setup, and usage.


