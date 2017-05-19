import pandas as pd
import pdb, re
import numpy as np
import scipy as sp
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
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
class SMSVectorizer(CountVectorizer):
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64):
        super(SMSVectorizer, self).__init__()

    def prepare_doc(self, doc):
        ques_list = ['?', 'what', 'were', 'where', 'who', 'when', 'why',
                     'which', 'how']
        for word in ques_list:
            doc = doc.replace(word, '_QUESTION')


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
X_train = train
y_train = train["Category"]
X_test = test
y_test = test["Category"]

def generate_train_test(X_train, X_test, y_train, y_test):
    Xtr, ytr = one_vs_all_classes(X_train, y_train, 0)
    Xte, yte = one_vs_all_classes(X_test, y_test, 0)
    return Xtr, Xte, ytr, yte

def misclassifications_class(model, Xte, yte):
    y_preds = model.predict(Xte["Message"])
    cats = ['Emergency', 'To-Do']
    sub_cats = ['E_Health', 'E_Personal', 'E_Fire', 'E_Police', 'TD_Urgent',
                'TD_Event', 'TD_General']
    misc = np.where(y_preds != yte)
    # select all misclassified in Emergency and To-DO
    df = Xte.iloc[misc]
    for sc in sub_cats:
        total = Xte[Xte["New Sub Category"] == sc].shape[0]
        miss = df[df["New Sub Category"] == sc].shape[0]
        print("%s Miss: %s/%s(%04.2f)%%"%(sc, miss, total, (miss/total)*100))

def analysis(model, X_test, y_test):
    clf = model.named_steps['classif']
    voc = model.named_steps['vect'].vocabulary_
    sr = feature_importances(clf, voc)
    print("top and bottom 10 features")
    print(sr[0:10])
    print(sr[-10:])
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(clsr(y_test, y_pred))

def feature_selection(model, X_train, y_train, X_test, y_test):
    clf = model.named_steps['classif']
    vect = model.named_steps['vect']
    sfm = SelectFromModel(clf, prefit=True, threshold = 0)
    n_features = sfm.transform(vect.transform(X_train)).shape[1]
    Xtr = None
    while n_features > 10:
        sfm.threshold += 0.1
        Xtr = sfm.transform(vect.transform(X_train))
        n_features = Xtr.shape[1]
    Xte = sfm.transform(vect.transform(X_test))
    clf = LogisticRegression()
    clf.fit(Xtr, y_train)
    clf.score(Xte, y_test)

def feature_importances(classif, vocab):
    coefs = None
    if hasattr(classif, 'coef_'):
        coefs = classif.coef_[0]
    else:
        coefs = classif.feature_importances_
    i_voc = {k:v for v, k in vocab.items()}
    feat = [(i_voc[idx], val) for idx, val in list(zip(range(0, len(coefs)), coefs))]
    return sorted(feat, key=lambda k:k[1], reverse=True)

def save_misc(mode, X_test, y_test):
    y_preds = model.predict(Xte)
    misc = np.where(y_preds != yte)
    miscXY = list(zip(Xte[misc], yte[misc]))
    fout1 = open('emergency_miss', 'w')
    fout2 = open('todo_miss', 'w')
    for line in miscXY:
        if line[1] == 1:
            fout1.write(line[0]+'\n')
        else:
            fout2.write(line[0]+'\n')
    fout1.close()
    fout2.close()

def generate_word_vectors(X, model, maxlen):
    vectors = np.empty((0,300))
    for line in X:
        idx = 1
        for word in line.strip().split():
            vec = np.zeros((1,300))
            try:
                vec += model[word]
                idx += 1
            except:
                pass
        vectors = np.append(vectors, vec / idx, axis=0)
    return vectors

def one_vs_all_classes(X, y, class_n):
    #y_n = np.empty(y.shape, dtype=int)
    #y_n[y==0] = 1
    #y_n[y==2] = 0
    #pos_num = len(y_n[y_n==1])
    xpos = X[y==0]
    xneg = X[y==2]
    #xneg = list(xneg[0:pos_num])
    return pd.concat([xpos, xneg]), np.array([1] * len(xpos) + [0] * len(xneg))

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
        if parameters:
            clf = GridSearchCV(model, parameters)
            clf.fit(X_train, y_train)
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            print()
            return clf
        else:
            model.fit(X_train, y_train)
            print(model.score(X_test, y_test))
        return model

    #word_vectors = KeyedVectors.load_word2vec_format('~/GoogleNews-vectors-negative300.bin', binary=True)  # C binary format
    # Label encode the targets
    labels = LabelEncoder()
    y_train = labels.fit_transform(y_train)
    y_test = labels.fit_transform(y_test)
    Xtr, Xte, ytr, yte = generate_train_test(X_train, X_test, y_train, y_test)
    # Begin evaluation
    if verbose: print("Building for evaluation")
    model = build(classifier, list(Xtr["Message"]), ytr, list(Xte["Message"]), yte) 
    misclassifications_class(model, Xte, yte)
    #feature_selection(model, Xtr, ytr, Xte, yte)
    #analysis(model, Xte, yte)
    #save_misc(model, Xte, yte)
    #vect = TfidfVectorizer(preprocessor=NLTKPreprocessor(),
    #            lowercase=True, stop_words='english', ngram_range=(1,2))
    #X_tr_mat = vect.fit_transform(X_train)
    #X_te_mat = vect.transform(X_test)
    #vector = np.append(X_train, X_test, axis=0)
    #vec_new = []
    #for line in vector:
    #    new_line = []
    #    for word in line.strip().split():
    #        if word not in stopwords.words('english'):
    #            new_line.append(word)
    #    vec_new.append(' '.join(new_line))
    #maxlen = len(max(vec_new, key=lambda k:len(k)).split())
    #vectors_train = generate_word_vectors(X_train, word_vectors, maxlen)
    #vectors_test = generate_word_vectors(X_test, word_vectors, maxlen)
    #X_tr_mat = sp.sparse.hstack((X_tr_mat, vectors_train), format='csr')
    #X_te_mat = sp.sparse.hstack((X_te_mat, vectors_test), format='csr')
    #classifier.fit(X_tr_mat, y_train)
    #print(classifier.score(X_te_mat, y_test))
    return model

#clf = svm.LinearSVC()
clf = LogisticRegression(penalty='l2')
#clf = RandomForestClassifier(n_estimators = 10, max_features='sqrt')
#clf = ExtraTreesClassifier()
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
