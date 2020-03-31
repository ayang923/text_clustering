import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from pprint import pprint

from nltk.corpus import stopwords
import nltk.tokenize as tk
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec

import logging
import sys

from encoder import infersent as inf

fmt = "[%(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=fmt)
logger = logging.getLogger(__name__)

def get_stopwords():
    with open('stopwords.txt', 'r') as file:
        stop_words = set(file.read().split("\n"))
    logger.debug(stop_words)
    
    return stop_words

def text_preprocessing(txt):
    #tokenizes into sentences
    sentences = tk.sent_tokenize(txt)
    logger.debug(sentences)
    
    #get stop_words
    stop_words = get_stopwords()
    
    #tokenizes into words
    lemmatizer = WordNetLemmatizer()
    tokenizer = tk.RegexpTokenizer(r'\w+')
    wordsList = [[lemmatizer.lemmatize(w.lower()) for w in tokenizer.tokenize(sentence) if not w in stop_words and not w.isdigit() and not len(w) <= 2] for sentence in sentences]
    
    #deletes unimportant sentences
    for i in range(len(wordsList)):
        if not wordsList[i]:
            sentences.pop(i)

    #deletes empty sentences
    wordsList = [x for x in wordsList if x]
    logger.debug(wordsList[0])
    
    return sentences, wordsList

def get_mean_vector(word2vec_model, words):
    # remove out-of-vocabulary words
    words = [word for word in words if word in word2vec_model.wv.vocab]
    logger.debug(words)
    if len(words) >= 1:
        return np.mean(word2vec_model[words], axis=0)
    else:
        return []

def wordvec_feature_extraction(wordsList):
    model = Word2Vec(wordsList, min_count=1)
    vectors = model[model.wv.vocab]
    logger.debug(vectors)
    
    return model, vectors

def sentvec_feature_extraction(wordsList):
    model = Sent2Vec(wordsList, size=100, min_count=1)

def kmeans(sentences, wordsList, k):
    wordvec_model, vectors = wordvec_feature_extraction(wordsList)
    
    kmeansmodel = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
    kmeansmodel.fit(vectors)
    
    order_centroids = kmeansmodel.cluster_centers_.argsort()[:, ::-1]
    terms = list(wordvec_model.wv.vocab)
    
    for i in range(k):
        logger.info("Cluster {}:".format(i))
        for ind in order_centroids[i, :10]:
            logger.info(' {}'.format(terms[ind]))
            
    predictions = [(i, kmeansmodel.predict([get_mean_vector(wordvec_model, wordsList[i])])[0]) for i in range(len(sentences))]
    predictions.sort(key=lambda tup: (tup[1], tup[0]))
    logger.info(predictions)
    return predictions

def infersent_embedding(wordsList, sentences):
    model = inf.infersent_train(sentences)
    
    embedding = model.encode(sentences)
    logger.info(embedding)
    
    kmeansmodel = KMeans(n_clusters=10, init='k-means++', max_iter=100, n_init=1)
    kmeansmodel.fit(embedding)
    
    logger.info(model.encode(["George Washington (February 22, 1732[b] – December 14, 1799) was an American political leader, military general, statesman, and Founding Father who served as the first president of the United States from 1789 to 1797."]))
    
    predictions = [(i, kmeansmodel.predict(model.encode([sentences[i]]))[0]) for i in range(len(sentences))]
    #predictions.sort(key=lambda tup: (tup[1], tup[0]))
    
    logger.info(predictions)
    
def main():
    
    #opens test text
    with open('test.txt', 'r') as file:
        testtxt = file.read().replace("\n", "")

    sentences, wordsList = text_preprocessing(testtxt)
    infersent_embedding(wordsList, sentences)
    
    predictions = kmeans(sentences, wordsList, 10)
    
    with open('result.txt', 'w') as file:
        current = 0
        for i, p in predictions:
            if p > current:
                file.write("\n\n")
                current += 1
                
            file.write(sentences[i])

if __name__ == "__main__":
    main()


