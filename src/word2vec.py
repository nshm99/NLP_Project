import re
import os
import numpy as np
from numpy import core
import pandas as pd
from time import time 
import logging 
import multiprocessing
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
# %matplotlib inline
 
import seaborn as sns
sns.set_style("darkgrid")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

def tsnescatterplot(model, word, list_names):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """
    arrays = np.empty((0, 300), dtype='f')
    word_labels = [word]
    color_list  = ['green']

    # adds the vector of the query word
    arrays = np.append(arrays, model.__getitem__([word]), axis=0)
    
    vocab = list(model.key_to_index)
    vocab = vocab[:50]
    X = model[vocab]
    for wrd in vocab:
        wrd_vector = model.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)
        
    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = PCA(n_components=2).fit_transform(arrays)
    
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
    
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
            
    plt.title('t-SNE visualization for {}'.format(word.title()))
    plt.savefig('../data/images/word_vectors.png')

def make_models():
    classes=['all',0,1]
    combined_data_word_broken = pd.read_csv('../data/process/combined_data_word_broken.csv')
    print(combined_data_word_broken['text'].shape)
    for c in classes:
        print("c = ",c,"_______________________________")
        sentences=[]
        for i in range(combined_data_word_broken.shape[0]):
            if c!='all' :
                if combined_data_word_broken['class'][i]==c:
                    sentences.append(combined_data_word_broken['text'][i].split())
            else:
                sentences.append(combined_data_word_broken['text'][i].split())

        cores = multiprocessing.cpu_count() # Count the number of cores in a computer

        print(cores)
        w2v_model = None

        w2v_model = Word2Vec(min_count=20,
                            vector_size=300,
                            workers=cores-3,
                            window=2,
                            sample=6e-5, 
                            alpha=0.03, 
                            min_alpha=0.0007, 
                            negative=20
                            )
        w2v_model.save(f"../models/word2vec_class_{c}.model")

        t = time()
        print(len(sentences),"__________++++++++++++++++++++++++++++++")

        w2v_model.build_vocab(sentences, progress_per=10000)

        print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

        t = time()

        w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

        print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
        word_vectors = w2v_model.wv
        word_vectors.save(f"../models/word2vec_class_{c}.wordvectors")

def plot_common(model_0,model_1,words):
    arrays = np.empty((0, 300), dtype='f')
    word_labels = []
    color_list  = []

    for wrd in words:
        wrd_vector = model_0.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)
    
    for wrd in words:
        wrd_vector = model_1.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('red')
        arrays = np.append(arrays, wrd_vector, axis=0)

    
    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = PCA(n_components=2).fit_transform(arrays)
    
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
    
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
            
    plt.title('t-SNE visualization')
    plt.savefig('../data/images/word_vectors_common.png')

def plot_similar(model_0,model_1,word):
    arrays = np.empty((0, 300), dtype='f')
    word_labels = [word,word]
    color_list  = ['green','red']

    similars_0 = model_0.most_similar([word])
    similars_1 = model_1.most_similar([word])

    arrays = np.append(arrays, model_0.__getitem__([word]), axis=0)
    arrays = np.append(arrays, model_1.__getitem__([word]), axis=0)

    for wrd in similars_0:
        wrd_vector = model_0.__getitem__([wrd[0]])
        word_labels.append(wrd[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)

    for wrd in similars_1:
        wrd_vector = model_1.__getitem__([wrd[0]])
        word_labels.append(wrd[0])
        color_list.append('orange')
        arrays = np.append(arrays, wrd_vector, axis=0)

    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = PCA(n_components=2).fit_transform(arrays)
    
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
    
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
            
    plt.title('t-SNE visualization')
    plt.savefig(f'../data/images/word_vectors_common_{word}.png')



make_models()
w2v_model_all = KeyedVectors.load("../models/word2vec_class_all.wordvectors", mmap='r')
w2v_model_0 = KeyedVectors.load("../models/word2vec_class_0.wordvectors", mmap='r')
w2v_model_1 = KeyedVectors.load("../models/word2vec_class_1.wordvectors", mmap='r')

plot_common(w2v_model_0,w2v_model_1,[
        "people","time","like","life","make","work","thing",'learn','right'])

plot_similar(w2v_model_0,w2v_model_1,'learn')

plot_similar(w2v_model_0,w2v_model_1,'right')
print("bisase:")
print("__1:")
print("male->secretary = female -> ",w2v_model_1.most_similar(positive=["female", "secretary"], negative=["male"], topn=3))
print("female->secretary = male -> ",w2v_model_1.most_similar(positive=["male", "secretary"], negative=["female"], topn=3))
print("__2:")
print("male->sexual = female -> ",w2v_model_1.most_similar(positive=["female", "sexual"], negative=["male"], topn=3))
print("female->sexual = male -> ",w2v_model_1.most_similar(positive=["male", "sexual"], negative=["female"], topn=3))

