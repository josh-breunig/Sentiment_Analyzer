# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 06:55:58 2022

@author: jbreu
"""

import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pattern


def normalize_data(doc, text_lemmatization=True, stopword_removal=True):
   # adjusting the stop word list
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.remove('no')
    stop_words.remove('but')
    stop_words.remove('not')
    lemmatizer = WordNetLemmatizer()
    
    normalized_text = []
    for text in doc:
        text = text.lower()
        text = text.strip()
        text = re.sub(r'[\r|\n|\r\n]', '', text) # removing html tags
        text = re.sub(r'[^a-zA-Z0-9]', ' ', text) # removing special characters
        text = re.sub(r'[0-9]', '', text) # removing numbers
        if text_lemmatization:
            tokens = word_tokenize(text)
            tokens = [token.strip() for token in tokens] 
            text = ' '.join([lemmatizer.lemmatize(w) for w in tokens])
        if stopword_removal:
            filtered_tokens = [token for token in tokens if token not in stop_words]
            text = ' '.join(filtered_tokens)
        # correct word lengthening
        pattern = re.compile(r'(.)\1{2,}')
        text = pattern.sub(r'\1\1', text)
        normalized_text.append(text)
    return normalized_text
