## Spam Filter App

This code is an implementation of a simple spam filter using a Multinomial Naive Bayes classifier and a Streamlit web application for the user interface. Let's break down each part of the code:

##### Importing libraries:

- import pandas as pd: Imports the Pandas library for data manipulation.
- from sklearn.naive_bayes import MultinomialNB: Imports the Multinomial Naive Bayes classifier from scikit-learn.
- from sklearn.model_selection import train_test_split: Imports the train_test_split function from scikit-learn to split the dataset into training and testing sets.
- from sklearn.feature_extraction.text import TfidfVectorizer: Imports the TfidfVectorizer class from scikit-learn to convert text data into TF-IDF vectors.
- import streamlit as st: Imports the Streamlit library for creating the web application.

##### Loading and preprocessing the dataset:

- Reads a CSV file named 'spam.csv' using Pandas.
- Extracts the text data ('v2') and the target variable ('v1') from the dataset.
- Splits the dataset into training and testing sets using train_test_split.

##### Vectorizing the text data:

- Creates a TfidfVectorizer object to convert the text data into TF-IDF vectors.
- Fits the vectorizer on the training text data (X_train) and transforms both the training and testing text data into TF-IDF vectors (X_train_vec and X_test_vec).

##### Training the Multinomial Naive Bayes model:

- Creates a Multinomial Naive Bayes classifier object.
- Fits the classifier on the training TF-IDF vectors and their corresponding target labels (y_train).

##### Streamlit User Interface:

- Sets the title of the Streamlit web application to 'Spam Filter'.
- Provides a text area for the user to input an email or message.
- Checks if the user has clicked the 'Predict' button.
- Preprocesses the input text by transforming it into a TF-IDF vector using the same vectorizer used for training.
- Makes a prediction using the trained Naive Bayes classifier.
- Displays the prediction ('spam' or 'ham') on the web application.
- Overall, this code creates a simple web application where users can input text, and the application predicts whether the text is spam or not based on a trained Multinomial Naive Bayes classifier.

### 1. Installation

- Clone the repository:
```bash
git clone https://github.com/TanishArya/SpamFilter.git
```

- Install the required Python packages:
```bash
pip install -r requirements.txt
```

- Usage
Run the Streamlit app:

```bash
streamlit run app.py
```
Open your web browser and go to http://localhost:8501 to access the application.

### 2. Built With
##### 1. Python - Programming language used

- Python is a high-level, interpreted programming language known for its simplicity and readability.
- It is widely used for various purposes, including web development, data analysis, artificial intelligence, scientific computing, automation, and more.
- Python's simplicity, readability, and extensive standard library make it a popular choice among developers for a wide range of applications.

##### 2. Streamlit - Web application framework

- Streamlit is an open-source Python library used to create interactive web applications for machine learning and data science projects.
- It allows developers to build web apps directly from Python scripts, without requiring knowledge of web development technologies such as HTML, CSS, or JavaScript.
- Streamlit provides a simple and intuitive API for creating user interfaces, making it easy to integrate machine learning models, visualizations, and data analysis tools into web applications.
- With Streamlit, developers can quickly prototype, iterate, and deploy web apps, accelerating the development process and enabling seamless communication of data insights.

##### 3. scikit-learn - Machine learning library

- Scikit-learn is a popular machine learning library in Python that provides simple and efficient tools for data mining, data analysis, and machine learning tasks.
- It is built on top of other scientific computing libraries in Python, such as NumPy, SciPy, and matplotlib, making it easy to integrate with existing data processing and visualization workflows.
- Scikit-learn offers a wide range of supervised and unsupervised learning algorithms, including classification, regression, clustering, dimensionality reduction, and model selection.
- The library is designed with ease of use and efficiency in mind, providing a consistent API and sensible default parameters for most algorithms.
- Scikit-learn also includes utilities for data preprocessing, feature extraction, model evaluation, and cross-validation, making it suitable for end-to-end machine learning workflows.
- Overall, scikit-learn is a powerful and versatile library that enables developers to build and deploy machine learning models with ease, making it a valuable tool for both beginners and experienced practitioners in the field of machine learning and data science.

### 3. Dataset
The dataset used for training the model is included as spam.csv. It contains text messages labeled as either 'spam' or 'ham' (not spam).

### 4. Library Uses

##### 1. Pandas (import pandas as pd):

- Pandas is a powerful data manipulation library in Python.
- It provides data structures and functions to work with structured data, primarily tabular data.
- In this code, Pandas is imported with the alias pd, which is a common convention.
- It is used to read the dataset stored in a CSV file and perform preprocessing tasks.

##### 2. scikit-learn (from sklearn.naive_bayes import MultinomialNB):

- Scikit-learn is a popular machine learning library in Python.
- It provides a wide range of tools for data mining, data analysis, and machine learning tasks.
- In this code, the MultinomialNB class is imported, which is used to implement the Multinomial Naive Bayes classifier.
- Naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features.

##### 3. scikit-learn (from sklearn.model_selection import train_test_split):

- This is another import from the scikit-learn library.
- It imports the train_test_split function, which is used to split the dataset into training and testing sets.
- This splitting is essential for evaluating the performance of the machine learning model on unseen data.

##### 4. scikit-learn (from sklearn.feature_extraction.text import TfidfVectorizer):

- Yet another import from scikit-learn.
- This time, it imports the TfidfVectorizer class, which is used to convert a collection of raw documents (text data) into a matrix of TF-IDF features.
- TF-IDF stands for Term Frequency-Inverse Document Frequency, and it is a numerical representation suitable for machine learning algorithms, particularly for text data.

##### 5. Streamlit (import streamlit as st):

- Streamlit is a Python library for building interactive web applications for machine learning and data science projects.
- It allows developers to create web apps directly from Python scripts, without needing to write HTML, CSS, or JavaScript.
- In this code, Streamlit is imported with the alias st, which is commonly used.
- It is used to create the user interface for the spam filter application, including text input and prediction display.
- These libraries collectively provide the necessary tools and functionalities to develop a spam filter application with a user-friendly interface, train a machine learning model, and make predictions on new text data.
