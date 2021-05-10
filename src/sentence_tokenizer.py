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

def sentence_tokenizer(blogs):
    totCount = 0
    tokenized = []
    for i in range(len(blogs)):
        text = blogs[i]
        sentence = sent_tokenize(text)
        totCount += len(sentence)
        tokenized.append(sentence)
    return tokenized,totCount

combined_data = pd.read_csv('../data/process/combined_data.csv')

combined_data_sentence_broken = pd.concat([combined_data],axis=0, ignore_index=True)
combined_data_sentence_broken.to_csv('../data/process/combined_data_sentence_broken.csv', index = False)

analyzed =[]
data ,count = sentence_tokenizer(combined_data["text"])
analyzed.append(["blog_count",len(combined_data)])
analyzed.append(["sen_count",count])
combined_data_sentence_broken["text"] = data

combined_data_sentence_broken.to_csv('../data/process/combined_data_sentence_broken.csv', index = False)
df = pd.DataFrame(analyzed)
df.to_csv('../data/process/analyze.csv', index = False)