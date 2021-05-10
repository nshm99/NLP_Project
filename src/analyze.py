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
from csv import writer

def word_count(blogs):
    total_count = 0
    total_count_0 = 0
    total_count_1 = 0
    unique_count = 0
    unique_text=""
    unique_text_0=""
    unique_text_1=""
    for i in range(len(blogs)):
        text =blogs["text"][i]
        total_count += len(text)
        unique_text +=  (text)
        if(blogs['class'][i]==0):
            total_count_0 += len(text)
            unique_text_0 +=  (text)
        elif(blogs['class'][i]==1):
            total_count_1 += len(text)
            unique_text_1 +=  (text)

    unique_text = unique_text.split()
    unique_text_0 = unique_text_0.split()
    unique_text_1 = unique_text_1.split()
    return total_count,len(set(unique_text)),len(set(unique_text_0)-set(unique_text_1)),\
        len(set(unique_text_1)-set(unique_text_0)),len(set(unique_text_0).intersection(set(unique_text_1))),\
           unique_text, unique_text_0,unique_text_1


def plot_most_used_words(category_string, data_series, palette):
    cvec = CountVectorizer(stop_words='english')
    cvec.fit(data_series)
    created_df = pd.DataFrame(cvec.transform(data_series).todense(),
                              columns=cvec.get_feature_names())
    total_words = created_df.sum(axis=0)
    
    top_10_words = total_words.sort_values(ascending = False).head(10)
    top_10_words_df = pd.DataFrame(top_10_words, columns = ["count"])

    sns.set_style("white")
    plt.figure(figsize = (12, 7), dpi=100)
    ax = sns.barplot(y= top_10_words_df.index, x="count", data=top_10_words_df, palette = palette)
    
    plt.xlabel("Count", fontsize=9)
    plt.ylabel('Common Words in {}'.format(category_string), fontsize=9)
    plt.yticks(rotation=-5)
    plt.show()

def plot_top_10(combined_data,text,text_0,text_1):
    motivational_posts = combined_data[combined_data["class"] ==0]["text"]
    nonMotivational_posts = combined_data[combined_data["class"] ==1]["text"]

    motiv_words_only = [x for x in text_0 if x not in text_1 ]
    nonMotiv_words_only = [x for x in text_1 if x not in text_0]

    plot_most_used_words("r/motivational unique Posts", motiv_words_only, palette="ocean_r")
    plot_most_used_words("r/non motivational unique Posts", nonMotiv_words_only, palette="ocean_r")

def relative_normilized__freq(text,text_0,text_1):
    common_words = [x for x in text_0 if x  in text_1 ]
    cvec = CountVectorizer(stop_words='english')
    cvec.fit(common_words)

    created_df = pd.DataFrame(cvec.transform(common_words).todense(),
                              columns=cvec.get_feature_names())
    total_words = created_df.sum(axis=0)
    top_10_words = total_words.sort_values(ascending = False).head(10)
    top_10_words_df = pd.DataFrame(top_10_words, columns = ["count"])

def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

def computeIDF(documents):
    import math
    N = len(documents)
    
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

def TF_IDF(text_0,text_1):

    uniqueWords = set(text_0).union(set(text_1))
    numOfWordsA = dict.fromkeys(uniqueWords, 0)
    for word in text_0:
        numOfWordsA[word] += 1
    numOfWordsB = dict.fromkeys(uniqueWords, 0)
    for word in text_1:
        numOfWordsB[word] += 1
    tfA = computeTF(numOfWordsA, text_0)
    tfB = computeTF(numOfWordsB, text_1)
    idfs = computeIDF([numOfWordsA, numOfWordsB])
    tfidfA = computeTFIDF(tfA, idfs)
    tfidfB = computeTFIDF(tfB, idfs)
    tfidfA = dict(Counter(tfidfA).most_common(10))
    tfidfB = dict(Counter(tfidfB).most_common(10))

    
    y = tfidfA.values()
    x = list(tfidfA.keys())
    plt.barh(x, y)
    plt.show()
    
    y = tfidfB.values()
    x = list(tfidfB.keys())
    plt.barh(x, y)
    plt.show()


    df = pd.DataFrame([tfidfA, tfidfB])
    df.to_csv('../data/process/TF-IDF.csv', index = False)

def frequency(text):
    # set_text = set(text)
    counts = Counter(text)

    labels, values = zip(*counts.items())

    # sort your values in descending order
    indSort = np.argsort(values)[::-1]

    # rearrange your data
    labels = np.array(labels)[indSort]
    values = np.array(values)[indSort]
    labels=labels[0:31]
    values = values[0:31]

    indexes = np.arange(len(labels))

    bar_width = 0.35

    plt.bar(indexes, values)

    # add labels
    plt.xticks(indexes + bar_width, labels)
    plt.show()


combined_data_word_broken = pd.read_csv('../data/process/combined_data_word_broken.csv')
combined_data = pd.read_csv('../data/process/combined_data.csv')
analyze = pd.read_csv('../data/process/analyze.csv')




analyzed =[]

tot_count,unique_count,unique_count_0,\
    unique_count_1,intersection_count,\
        text,text_0,text_1 = word_count(combined_data_word_broken)

analyzed.append(["tot_word_count",tot_count])
analyzed.append(["tot_unique_word_count",unique_count])
analyzed.append(["tot_unique_word_count_1",unique_count_0])
analyzed.append(["tot_unique_word_count_0",unique_count_1])
analyzed.append(["tot_intersection_word_count",intersection_count])


with open('../data/process/analyze.csv', 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    for a in analyzed:
        csv_writer.writerow(a)

#plot
plot_top_10(combined_data,text,text_0,text_1)
# relative_normilized__freq(text,text_0,text_1)
TF_IDF(text_0,text_1)

frequency(text)

