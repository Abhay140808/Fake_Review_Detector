# Fake_Review_Detector

This project implements a machine learning pipeline for detecting fake reviews using natural language processing and a supervised classification approach. The model leverages a cleaned dataset of reviews, performs extensive text preprocessing, applies TF-IDF feature engineering, and classifies reviews with a Random Forest algorithm.

## Project Overview
The primary objective is to distinguish between genuine and fake (computer-generated) product reviews using text data. The dataset, sourced from Kaggle, contains labeled review texts which are cleaned and preprocessed prior to model training.

## Data Preprocessing and Feature Engineering
- Non-essential columns are removed, and only valid entries with known labels and non-empty review texts are retained.

- Text data undergoes normalization: lowercasing, removal of non-alphabetic characters, tokenization, and stopword elimination.

- Cleaned reviews are converted into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency) with a feature limit of 5,000.

- This results in a robust word vector representation suitable for machine learning tasks.

## Model Architecture
- After preprocessing, the data is split into training (80%) and test sets (20%).

- The model employs a Random Forest classifier implemented using scikit-learn, with 100 estimators and a fixed random state for reproducibility.

- The classifier is trained to distinguish between real (label: 0) and fake (label: 1) reviews, learning patterns indicative of authenticity or fraudulence.

## Evaluation
- The model achieves an accuracy of approximately 85% on the test set.

- Detailed metrics reveal balanced precision, recall, and F1-scores (all about 0.85) for both classes, demonstrating consistent detection ability for real and fake reviews.

## Usage
- This project provides a reproducible and scalable solution for fake review detection, emphasizing explainable preprocessing and rigorous evaluation. The notebook serves as an end-to-end demonstration suitable for further experimentation, deployment, or integration into review moderation systems.
