# Spam Email Detection Using TensorFlow

This project demonstrates how to build a deep learning model to classify emails as **Spam** or **Ham** (Not Spam) using TensorFlow and Python. The workflow covers data loading, preprocessing, balancing, visualization, model building, training, and evaluation.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Steps](#project-steps)
  - [1. Import Required Libraries](#1-import-required-libraries)
  - [2. Load the Dataset](#2-load-the-dataset)
  - [3. Visualize Class Distribution](#3-visualize-class-distribution)
  - [4. Balance the Dataset](#4-balance-the-dataset)
  - [5. Clean the Text](#5-clean-the-text)
  - [6. Visualize with Word Cloud](#6-visualize-with-word-cloud)
  - [7. Tokenization and Padding](#7-tokenization-and-padding)
  - [8. Define the Model](#8-define-the-model)
  - [9. Train the Model](#9-train-the-model)
  - [10. Evaluate the Model](#10-evaluate-the-model)
- [Results](#results)
- [How to Run](#how-to-run)
- [Requirements](#requirements)

---

## Project Overview
Spam detection is a classic text classification problem. This project uses a deep learning approach (LSTM) to automatically classify emails as spam or not spam.

## Dataset
- **File:** `spam_ham_dataset.csv`
- Contains email messages labeled as `spam` or `ham`.

## Project Steps

### 1. Import Required Libraries
All necessary libraries for data manipulation, visualization, text processing, and deep learning are imported, including `pandas`, `seaborn`, `nltk`, `wordcloud`, and `tensorflow`.

### 2. Load the Dataset
The dataset is loaded into a pandas DataFrame and the first few rows are displayed to understand its structure.

### 3. Visualize Class Distribution
A countplot is used to visualize the number of spam and ham emails, with data labels on each bar for clarity.

### 4. Balance the Dataset
The dataset is imbalanced (more ham than spam). To address this:
- The ham and spam emails are separated.
- The majority class (ham) is downsampled to match the number of spam emails.
- The balanced dataset is visualized again.

### 5. Clean the Text
Text cleaning steps include:
- Removing the 'Subject:' prefix.
- Removing punctuation.
- Removing stopwords using NLTK.

### 6. Visualize with Word Cloud
Word clouds are generated for both spam and ham emails to visualize the most frequent words in each class.

### 7. Tokenization and Padding
- The cleaned text is split into training and test sets.
- Text is tokenized (converted to sequences of integers).
- Sequences are padded to ensure uniform length for model input.
- Labels are converted to numeric (0 for ham, 1 for spam).

### 8. Define the Model
A Sequential model is built with:
- An Embedding layer to learn word representations.
- An LSTM layer to capture sequence patterns.
- A Dense layer for feature extraction.
- An output layer with sigmoid activation for binary classification.

### 9. Train the Model
- The model is compiled with binary cross-entropy loss and the Adam optimizer.
- Early stopping and learning rate reduction callbacks are used.
- The model is trained on the training data and validated on the test data.

### 10. Evaluate the Model
- The model's performance is evaluated on the test set.
- Accuracy and loss are reported.
- Training and validation accuracy are plotted over epochs.

## Results
- The model achieves good accuracy in classifying emails as spam or ham.
- Visualization and evaluation steps help interpret the model's performance.

## How to Run
1. Clone this repository.
2. Install the required packages (see below).
3. Place `spam_ham_dataset.csv` in the project directory.
4. Open and run `main.ipynb` step by step in Jupyter Notebook or VS Code.

## Requirements
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- nltk
- wordcloud
- tensorflow
- scikit-learn

Install requirements with:
```bash
pip install pandas numpy matplotlib seaborn nltk wordcloud tensorflow scikit-learn
```

---

**Author:**
- Your Name

**License:**
- MIT
