import os
import io
import tarfile
import urllib.request
from collections import Counter
from math import log
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tkinter as tk

# Download the spam and ham datasets
url = 'https://spamassassin.apache.org/old/publiccorpus/20021010_easy_ham.tar.bz2'
urllib.request.urlretrieve(url, 'easy_ham.tar.bz2')
tar = tarfile.open('easy_ham.tar.bz2')
tar.extractall()
tar.close()

url = 'https://spamassassin.apache.org/old/publiccorpus/20021010_spam.tar.bz2'
urllib.request.urlretrieve(url, 'spam.tar.bz2')
tar = tarfile.open('spam.tar.bz2')
tar.extractall()
tar.close()

# Define some constants
SPAM = 'spam'
HAM = 'ham'
TRAINING_RATIO = 0.8

# Define a function to read the emails from a directory
def read_emails(directory):
    emails = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        with io.open(path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            emails.append(content)
    return emails

# Read the spam and ham emails
spam_emails = read_emails('spam')
ham_emails = read_emails('ham')

# Split the data into training and testing sets
spam_cutoff = int(len(spam_emails) * TRAINING_RATIO)
ham_cutoff = int(len(ham_emails) * TRAINING_RATIO)
train_spam = spam_emails[:spam_cutoff]
test_spam = spam_emails[spam_cutoff:]
train_ham = ham_emails[:ham_cutoff]
test_ham = ham_emails[ham_cutoff:]

# Convert the text into bag-of-words vectors
vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_spam + train_ham)
X_test = vectorizer.transform(test_spam + test_ham)

# Create the target labels
y_train = [SPAM] * len(train_spam) + [HAM] * len(train_ham)
y_test = [SPAM] * len(test_spam) + [HAM] * len(test_ham)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)