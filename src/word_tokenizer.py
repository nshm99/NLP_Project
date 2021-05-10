import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
import re
import os
from string import punctuation
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import seaborn as sns
import math
from textblob import TextBlob as tb
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter


def process(blogs):
    processed = []
    for i in range(len(blogs)):
        text = blogs[i]
        
        lemmatizer = WordNetLemmatizer()
        tokenizer = RegexpTokenizer(r'(\w+)')
        
        text = re.sub(f"[{re.escape(punctuation)}]", "",text)
        words_only = tokenizer.tokenize(text)
        words_only_lem = [lemmatizer.lemmatize(i) for i in words_only]
        words_without_stop = [i for i in words_only_lem if i not in stopwords.words("english")]
        long_string_clean = " ".join(word for word in words_without_stop)
        processed.append(long_string_clean)

    return processed

combined_data = pd.read_csv('../data/process/combined_data.csv')
combined_data_word_broken = pd.concat([combined_data],axis=0, ignore_index=True)
combined_data_word_broken.to_csv('../data/process/combined_data_word_broken.csv', index = False)

combined_data_word_broken["text"] = process(combined_data["text"])

combined_data_word_broken.to_csv('../data/process/combined_data_word_broken.csv', index = False)
