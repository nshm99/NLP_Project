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
# import heapq
from collections import Counter

def remove_date(blogs):
    removed = []
    for i in range(len(blogs)):
        text = blogs[i].lower()
        x = text.find("min read")
        text = text[x+10:] 
        tokens = text.split()
        clean_tokens = [t for t in tokens if len(t) > 2]
        text = " ".join(clean_tokens)
        removed.append(text)
    return removed


motivational_blogs = pd.read_csv('../data/raw/motivational.csv')
nonMotivational_blogs = pd.read_csv('../data/raw/nonMotivational.csv')

combined_data = pd.concat([motivational_blogs,nonMotivational_blogs],axis=0, ignore_index=True)
combined_data.to_csv('../data/process/combined_data.csv', index = False)

combined_data["text"] = remove_date(combined_data["text"])

combined_data.to_csv('../data/process/combined_data.csv', index = False)