#!/usr/bin/env python3
from sklearn.feature_extraction.text import CountVectorizer

# Input sentences
sentences = [
'are' 
'awesome' 
'beautiful' 
'cake' 
'children' 
'future' 
'good'
'grandchildren' 
'holberton' 
'is' 
'learning' 
'life' 
'machine'
'nlp' 
'no'
'not' 
'one' 
'our' 
'said'
'school'
'that' 
'the' 
'very'
'was'
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
