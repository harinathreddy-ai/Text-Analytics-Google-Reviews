# Google Reviews Rating Prediction

## Project Overview
This project focuses on predicting user ratings (1-5 stars) from Google review text using various Natural Language Processing (NLP) techniques and machine learning models. The goal is to build an accurate predictive model and gain insights into the textual patterns associated with different ratings.

## Table of Contents
1.  [Data Exploration and Visualization](#1-data-exploration-and-visualization)
2.  [Text Processing & Normalization](#2-text-processing--normalization)
3.  [Vector Space Model & Feature Representation](#3-vector-space-model--feature-representation)
4.  [Model Training, Selection & Hyperparameter Tuning](#4-model-training-selection--hyperparameter-tuning)
5.  [LSTM Sequence Model](#5-lstm-sequence-model)
6.  [BERT Model](#6-bert-model)
7.  [Topic Modelling of High and Low Ratings](#7-topic-modelling-of-high-and-low-ratings)

## 1. Data Exploration and Visualization
-   Loading and inspecting `train.csv` and `test.csv` datasets.
-   Analyzing the distribution of ratings, identifying class imbalance.
-   Visualizing word counts per rating and generating word clouds for 1-star and 5-star reviews to understand textual patterns.

## 2. Text Processing & Normalization
-   Implementation of a text preprocessing pipeline:
    -   Lowercasing and removal of punctuation/numbers.
    -   Stopword removal.
    -   Lemmatization (chosen over stemming for better performance).
-   Comparison of token counts at various stages of preprocessing.

## 3. Vector Space Model & Feature Representation
-   Using TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction.
-   Experimentation with different N-gram ranges (unigram, bigram, trigram) to find optimal representation.
-   Analysis of vocabulary size impact on model accuracy.

## 4. Model Training, Selection & Hyperparameter Tuning
-   Evaluation of traditional machine learning models using 5-fold cross-validation:
    -   Logistic Regression
    -   Multinomial Naive Bayes
    -   Linear SVM
-   Hyperparameter tuning for Logistic Regression using `GridSearchCV` to optimize regularization (`C` parameter).
-   Performance assessment using classification reports and confusion matrices.

## 5. LSTM Sequence Model
-   Implementation of a Bidirectional LSTM (BiLSTM) model using TensorFlow/Keras.
-   Tokenization and padding of text sequences.
-   Training with Early Stopping to prevent overfitting.
-   Evaluation with classification report and confusion matrix.

## 6. BERT Model
-   Fine-tuning a pre-trained BERT model (`nlptown/bert-base-multilingual-uncased-sentiment`) for sentiment classification.
-   Preparation of custom PyTorch datasets and DataLoaders.
-   Training the BERT model with AdamW optimizer and linear scheduler.
-   Evaluation using F1-score, classification report, and confusion matrix.

## 7. Topic Modelling of High and Low Ratings
-   Application of Latent Dirichlet Allocation (LDA) to identify key topics in 1-star (negative) and 5-star (positive) reviews.
-   Visualization of LDA topics using `pyLDAvis`.

## Technologies Used
-   Python
-   `pandas` for data manipulation
-   `nltk` for text preprocessing
-   `scikit-learn` for traditional ML models and evaluation
-   `tensorflow`/`keras` for LSTM model
-   `transformers` (Hugging Face) and `torch` for BERT model
-   `matplotlib`, `seaborn`, `wordcloud` for data visualization
-   `gensim`, `pyLDAvis` for topic modeling
