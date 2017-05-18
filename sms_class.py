import pandas as pd
import pdb, re
import numpy as np
from nltk.stem.porter import *
from nltk import word_tokenize
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from gensim.models.keyedvectors import KeyedVectors

class NLTKPreprocessor(object):
    def __init__(self):
        self.stemmer = PorterStemmer()
    def preprocess(self, word):
        w = self.stemmer.stem(word)
        w = re.sub('\d+', '_NUMBER', w)
        w = re.sub(',\[\]\(\)\.', ' ', w)
        w = re.sub('\s+', ' ', w)
        w = re.sub('\'', '', w)
        return w
    def __call__(self, doc):
        words = [self.preprocess(t) for t in word_tokenize(doc)]
        return " ".join(words)


train = pd.read_csv("training.csv", header=0, \
                    delimiter="\t", quoting=3)
test = pd.read_csv("testing.csv", header=0, \
                    delimiter="\t", quoting=3)
X_train = train["Message"]
y_train = train["Category"]
X_test = test["Message"]
y_test = test["Category"]

def feature_importances(classif, vocab):
    coefs = None
    if hasattr(classif, 'coef_'):
        coefs = classif.coef_[0]
    else:
        coefs = classif.feature_importances_
    i_voc = {k:v for v, k in vocab.items()}
    feat = [(i_voc[idx], val) for idx, val in list(zip(range(0, len(coefs)), coefs))]
    return sorted(feat, key=lambda k:k[1], reverse=True)

#def generate_word_vectors(X, model, maxlen):
#    vectors = np.array((0,maxlen*300))
#    for line in X:
#        line_vec = np.array([])
#        for word in line.strip().split():
#            vec = np.zeros((1,300))
#            try:
#                vec = model[word]
#            except:
#                pass
#            np.append(line_vec, vec)
#        while line_vec.shape[0] < maxlen * 300:
#            np.append(line_vec, np.zeros((1,300)))
#        np.append(vectors, line_vec, axis=0)
#    return vectors

def one_vs_all_classes(X, y, class_n):
    #y_n = np.empty(y.shape, dtype=int)
    #y_n[y==0] = 1
    #y_n[y==2] = 0
    #pos_num = len(y_n[y_n==1])
    xpos = list(X[y==2])
    xneg = list(X[y==1])
    #xneg = list(xneg[0:pos_num])
    return np.array(xpos+xneg), np.array([1] * len(xpos) + [0] * len(xneg))

def build_and_evaluate(X_train, y_train, X_test, y_test, parameters=None,
    classifier=svm.LinearSVC(), outpath=None, verbose=True):

    def build(classifier, X_train, y_train, X_test, y_test):
        """
        Inner build function that builds a single model.
        """
        #if isinstance(classifier, type):
        #    classifier = classifier()

        model = Pipeline([
            ('vect', TfidfVectorizer(preprocessor=NLTKPreprocessor(),
                lowercase=True, stop_words='english', ngram_range=(1,2))
            ),
            ('classif', classifier),
        ])
        #kf = KFold(n_splits=5, shuffle=True)
        #for train, test in kf.split(X):
        #    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        #    model.fit(X_train, y_train)
        #    print(model.score(X_test, y_test))
        if parameters:
            clf = GridSearchCV(model, parameters)
            clf.fit(X_train, y_train)
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            print()
            #y_true, y_pred = y_test, clf.predict(X_test)
            #print(clsr(y_true, y_pred))
        else:
            #ovr = OneVsRestClassifier(model)
            #ovr.fit(X_train, y_train)
            model.fit(X_train, y_train)
            print(model.score(X_test, y_test))
        return model

    word_vectors = KeyedVectors.load_word2vec_format('/run/media/sumit/linux/POST/GoogleNews-vectors-negative300.bin.gz', binary=True)  # C binary format
    # Label encode the targets
    labels = LabelEncoder()
    y_train = labels.fit_transform(y_train)
    y_test = labels.fit_transform(y_test)
    # For class one
    Xtr, ytr = one_vs_all_classes(X_train, y_train, 0)
    Xte, yte = one_vs_all_classes(X_test, y_test, 0)
    # Begin evaluation
    if verbose: print("Building for evaluation")
    #model = build(classifier, X_train, y_train, X_test, y_test)
    #svm = model.named_steps['classif']
    #voc = model.named_steps['vect'].vocabulary_
    #sr = feature_importances(svm, voc)
    vect = TfidfVectorizer(preprocessor=NLTKPreprocessor(),
                lowercase=True, stop_words='english', ngram_range=(1,2))
    X_tr_mat = vect.fit_transform(X_train)
    X_te_mat = vect.transform(X_test)
    vector = np.append(X_train, X_test, axis=0)
    maxlen = len(max(vector, key=lambda k:len(k)).split())
    #vectors = generate_word_vectors(X_train, word_vectors, maxlen)
    pdb.set_trace()
    print("top 10 words")
    print(sr[0:10])
    print("bottom 10 words")
    print(sr[-10:])
    #pdb.set_trace()
    if verbose:
        #print("Evaluation model fit in {:0.3f} seconds".format(secs))
        print("Classification Report:\n")
    y_pred = model.predict(Xte)
    #y_pred_prob = model.predict_proba(X_test)
    #y1 = y_pred_prob[:,0]
    #y_pred[y_pred==0] = 2
    #y_pred[y_pred==1] = 0
    #y_pred[(y1>0.45) & (y1<0.7)] = 1
    print(confusion_matrix(yte, y_pred))
    print(clsr(yte, y_pred))
    return model

#clf = RandomForestClassifier(n_estimators=10)
clf = svm.LinearSVC()
clf = LogisticRegression(penalty='l2' )
#clf = RandomForestClassifier()
#params = {
#    'vect__max_df': (0.5, 0.75, 1.0),
#    'vect__ngram_range': ((1, 1), (1, 2)),
#    'classif__max_features': ('sqrt', 'log2')
#}
params = { 'vect__ngram_range': ((1, 1), (1, 2)),
    'classif__kernel': ('rbf', 'linear'),
    'classif__C': (10,100),
}
build_and_evaluate(X_train, y_train, X_test, y_test, None, clf)
