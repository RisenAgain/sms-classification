# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics import classification_report as clsr
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from nltk import word_tokenize
import scipy as sp
import re
import numpy as np

class NLTKPreprocessor(object):
    def __init__(self, stem=True):
        self.stemmer = PorterStemmer()
        self.stem = stem
    def preprocess(self, word):
        w = word
        if self.stem:
            w = self.stemmer.stem(word)
        w = w.lower()
        w = re.sub(',\[\]\(\)\.-', ' ', w)
        w = re.sub('\'', ' ', w)
        w = re.sub('\s+', ' ', w)
        w = re.sub('\d+', '_NUMBER', w)
        return w
    def __call__(self, doc):
        words = [self.preprocess(t) for t in word_tokenize(doc)]
        return " ".join(words)


def generate_word_vectors_concat(model, words, X):
    wv_list = []
    for w in words:
        try:
            v = model[w]
            wv_list.append(v)
        except:
            pass
    vectors = np.empty((0,300*len(wv_list)))
    for x in X:
        sent = x.lower().split()
        vec_line = np.array([])
        for w in wv_list:
            vec = np.zeros((1,300))
            try:
                if w in sent:
                    vec = model[w]
            except:
                pass
            vec_line = np.append(vec_line, vec)
        vec_line = vec_line.reshape((1, 300*len(wv_list)))
        vectors = np.append(vectors, vec_line, axis=0)
    return vectors

def generate_word_vectors_avg(model, words, X):
    wv_list = []
    for w in words:
        try:
            v = model[w]
            wv_list.append(v)
        except:
            pass
    vector = np.empty((0,300))
    for x in X:
        sent = x.lower().split()
        vec = np.zeros((1,300))
        idx = 0
        for w in wv_list:
            if w in sent:
                vec += model[w]
                idx += 1
        if idx == 0:
            idx = 1
        #pdb.set_trace()
        vector = np.append(vector, vec / idx, axis=0)
    return vector


def extract_full_words(prep, words, vocab):
    wordlist = []
    for w in vocab:
        if prep.stemmer.stem(w) in words:
            wordlist.append(w)
    return wordlist


def feature_importances(classif, vocab):
    feats = []
    if isinstance(classif, OneVsRestClassifier):
        for cls in classif.estimators_:
            feats.append(feature_importances(cls, vocab))
    else:
        coefs = None
        if hasattr(classif, 'coef_'):
            coefs = classif.coef_[0]
        elif hasattr(classif, 'feature_importances_'):
            coefs = classif.feature_importances_
        else:
            return
        i_voc = {k:v for v, k in vocab.items()}
        feat = [(i_voc[idx], val, idx) for idx, val in list(zip(range(0, len(coefs)), coefs))]
        feats = sorted(feat, key=lambda k:k[1], reverse=True)
        # top and bottom 7
        feats = feats[0:7] + feats[-7:]
    return feats


def analysis(model, X_test, y_test):
    clf = model.named_steps['classif']
    voc = model.named_steps['vect'].vocabulary_

    y_pred = model.predict(X_test["Message"])
    print(confusion_matrix(y_test, y_pred))
    print(clsr(y_test, y_pred))

    sr = feature_importances(clf, voc)
    if sr is not None:
        print("top and bottom 7 features")
        print(sr)
        return sr


def with_wv(model, Xtr, ytr, Xte, yte):
    """
    Experimental method to test the word vector feature
    """
    word_vectors = KeyedVectors.load_word2vec_format('~/GoogleNews-vectors-negative300.bin', binary=True)  # C binary format
    prep = NLTKPreprocessor(stem=False)
    vect = TfidfVectorizer(preprocessor=prep,
                lowercase=True, stop_words='english', ngram_range=(1,2))
    X_tr_mat = vect.fit_transform(Xtr["Message"])
    X_te_mat = vect.transform(Xte["Message"])
    vocab = vect.get_feature_names()
    imp_words = list(zip(*analysis(model, Xte, yte)))
    #imp_indices = list(imp_words[2])
    #vocab = np.array(vocab)
    words = list(imp_words[0])
    imp_names = extract_full_words(prep, words, vocab)
    X_tr_mat = model.named_steps['vect'].transform(Xtr["Message"])
    X_te_mat = model.named_steps['vect'].transform(Xte["Message"])
    
    vectors_train = generate_word_vectors_avg(word_vectors, imp_names, Xtr["Message"])
    vectors_test = generate_word_vectors_avg(word_vectors, imp_names, Xte["Message"])

    X_tr_mat = sp.sparse.hstack((X_tr_mat, vectors_train), format='csr')
    X_te_mat = sp.sparse.hstack((X_te_mat, vectors_test), format='csr')
    cls = model.named_steps['classif']
    cls.fit(X_tr_mat, ytr)
    y_pred = (cls.predict(X_te_mat))
    print(clsr(yte, y_pred))
