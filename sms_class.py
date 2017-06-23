# -*- coding: utf-8 -*-
import pandas as pd
import pdb, sys, traceback
import numpy as np
import scipy as sp
from nltk.stem.porter import *
from sklearn import svm
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from features import gen_msg_features
from lib import NLTKPreprocessor, analysis, feature_importances
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import argparse


cat_map = {1: 'Emergency', 2: 'To-Do', 3: 'General'}
clf_map = {
            1: LogisticRegression(penalty='l2', C=10),
            2: RandomForestClassifier(n_estimators = 100, max_features='sqrt'),
            3: GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
        }

def process_data(train, test, val, args):
    filter_col = 'Category'
    if args.level == 2:
        filter_list = [cat_map[args.top_level]]
        train = train[train["Category"].isin(filter_list)]
        test = test[test["Category"].isin(filter_list)]
        val = val[val["Category"].isin(filter_list)]
        filter_col = 'Sub-Category'
    
    X_train = train
    y_train = train[filter_col]
    X_test = test
    y_test = test[filter_col]
    X_val = val
    y_val = val[filter_col]
    
    build_and_evaluate(X_train, y_train, X_test, y_test, X_val, y_val , args,
        clf_map[args.clf], None)


def plot_feat(feats, labels, encoder, y, title):
    values = []
    total = []
    for idx1, id1 in enumerate(labels):
        f_arr = []
        for idx2, id2 in enumerate(encoder.classes_):
            indices = y==idx2
            f_arr.append(np.mean(feats[indices, idx1], axis=0))
        values.append(f_arr)
    for idx, id in enumerate(encoder.classes_):
        indices = y[y==idx]
        total.append(len(indices))
    #values.append(total)
    # Plot
    plt.figure()
    plt.title(title)
    index = np.arange(len(encoder.classes_))
    colors = ['r', 'b', 'g', 'y', 'c', 'm', '#c21211', '#000000', '#00aa11',
              '#22cc22', '#bfe23e']
    #labels = np.append(labels,'total')
    for idx1, id1 in enumerate(labels):
        plt.bar(index+0.05*idx1, values[idx1], width=0.05, color=colors[idx1],
                align='center', label=id1)
    x_axis = np.lib.pad(index, (1,1), 'constant',\
                        constant_values=(-0.4,len(index)-1+0.4))
    plt.plot(x_axis, [0]*(len(index)+2), color='#000000')
    plt.xticks(index, encoder.classes_)
    plt.ylabel('Average Feature value')
    plt.xlabel('Category')
    plt.legend()
    plt.show()


def plot_surf(clf, X, y):
    # http://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_multinomial.html#sphx-glr-auto-examples-linear-model-plot-logistic-multinomial-py
    h = 0.1
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    #pdb.set_trace()
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.title("Decision surface of LogisticRegression")
    plt.axis('tight')

    # Plot training points
    colors = "bry"
    for i, color in zip(clf.classes_, colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired)

    plt.show()

def generate_train_test(X_train, X_test, y_train, y_test):
    Xtr, ytr = one_vs_all_classes(X_train, y_train, 2)
    Xte, yte = one_vs_all_classes(X_test, y_test, 2)
    return Xtr, Xte, ytr, yte

def misclassifications_class(model, Xte, yte, msgs, label_enc, level, wtf=False):
    y_preds = model.predict(Xte)
    misc = np.where(y_preds != yte)
    # select all misclassified 
    df = msgs.iloc[misc]
    misc_labels = label_enc.inverse_transform(y_preds)[misc]
    index = 'Category'
    if level == 1:
        index = 'Sub-Category'
    for sc in label_enc.classes_:
        total = msgs[msgs[index] == sc].shape[0]
        miss = df[df[index] == sc].shape[0]
        print("%s Miss: %s/%s(%04.2f)%%"%(sc, miss, total, (miss/max(total,1))*100))
        if wtf:
            miss_data = df[df[index] == sc]
            miss_data['Classifier Label'] = misc_labels[df[index] == sc]
            miss_data.to_csv(sc.lower(), sep = '\t')

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


def one_vs_all_classes(X, y, class_n):
    return X.iloc[y!=class_n], y[y!=class_n]


def build_and_evaluate(X_train, y_train, X_test, y_test, X_val, y_val,
    args, classifier=svm.LinearSVC(), parameters = None):

    def build(classifier, X_train, y_train, X_test, y_test):
        """
        Inner build function that builds a single model using only ngrams.
        """
        model = Pipeline([
            ('vect',
             TfidfVectorizer(preprocessor=NLTKPreprocessor(stem=True),lowercase=True,
                             stop_words=None,ngram_range=(1,2),
                             sublinear_tf=True)),
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
            X = np.append(X_train, X_test, axis=0)
            y = np.append(y_train, y_test, axis=0)
            kf = KFold(n_splits=4, shuffle=True)
            for train, test in kf.split(X):
                model.fit(X[train], y[train])
                print(model.score(X[test], y[test]))
            model.fit(X_train, y_train)
            print(model.score(X_test, y_test))
        return model

    # Label encode the targets
    labels = LabelEncoder()
    y_train = labels.fit_transform(y_train)
    print(labels.classes_)
    try:
        y_test = labels.fit_transform(y_test)
        y_val = labels.fit_transform(y_val)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
    
    Xtr, Xte, ytr, yte = X_train, X_test, y_train, y_test
    vectors_train = np.array(gen_msg_features(Xtr["Message"]))
    vectors_test = np.array(gen_msg_features(Xte["Message"]))
    vectors_val = np.array(gen_msg_features(X_val["Message"]))
    
    #model = build(classifier, list(Xtr["Message"]), ytr, list(Xte["Message"]), yte)
    #analysis(model, Xte, yte)
    
    prep = NLTKPreprocessor(stem=True)
    vect = TfidfVectorizer(preprocessor=prep,
                lowercase=True, stop_words=None, ngram_range=(1,2))
    X_tr_mat = vect.fit_transform(Xtr["Message"])
    X_te_mat = vect.transform(Xte["Message"])
    X_val_mat = vect.transform(X_val["Message"])
    
    X_tr_mat = sp.sparse.hstack((X_tr_mat, vectors_train[1]), format='csr')
    X_te_mat = sp.sparse.hstack((X_te_mat, vectors_test[1]), format='csr')
    X_val_mat = sp.sparse.hstack((X_val_mat, vectors_val[1]), format='csr')
    
    cls = classifier
    cls.fit(X_tr_mat, ytr)
    
    y_pred = (cls.predict(X_val_mat))
    print(confusion_matrix(y_val, y_pred))
    print(clsr(y_val, y_pred))
    
    #misclassifications_class(cls, X_val_mat, y_val, X_val,
    #                         labels, 1, True)
    if args.with_graph:
        plot_feat(vectors_val[1], vectors_val[0], labels, y_val,
                    'Average Feature value for each class')


if __name__ == "__main__":
    train = pd.read_csv("trainFile", header=0, \
                        delimiter="\t", quoting=3)
    test = pd.read_csv("tuneFile", header=0, \
                        delimiter="\t", quoting=3)
    val = pd.read_csv("testFile", delimiter="\t")
    
    parser = argparse.ArgumentParser(description='Classify SMS messages')
    parser.add_argument('--level', type=int, choices=(1, 2), default=1,
                        help="Level to classify on")
    parser.add_argument('--clf', type=int, choices=(1, 2, 3), default=1,
                        help="Classifier: Logistic Regression, Random Forest,\
                        GBC")
    parser.add_argument('--top_level', type=int, choices=(1, 2, 3), default=1,
                        help="Top level category to filter if level 2,\
                        Emergency, To-Do, General")
    parser.add_argument('--with_graph', action='store_true',
                        help="Whether to print feature graph")
    parser.parse_args()
    args = parser.parse_args()
    process_data(train, test, val, args)
