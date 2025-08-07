#!/usr/bin/env python3
from sklearn.feature_extraction.text import CountVectorizer

# Input sentences
sentences = [
    "Holberton school is awesome",
    "Holberton school is the future",
    "Holberton school is the best school",
    "Holberton school is not the best school",
    "My children are learning at Holberton school",
    "My grandchildren are learning at Holberton school",
    "My children are not learning at Holberton school",
    "My grandchildren are not learning at Holberton school"
]

# Create CountVectorizer with custom token pattern to exclude 's'
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')

# Fit and transform the sentences
X = vectorizer.fit_transform(sentences).toarray()

# Get the vocabulary list
vocab = vectorizer.get_feature_names_out()

# Print the matrix without commas
for row in X:
    print(' '.join(str(x) for x in row))

# Print the vocabulary without commas
print(' '.join(vocab))
