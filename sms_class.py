import pandas as pd
import pdb, sys, traceback
import re, csv
import numpy as np
import scipy as sp
import logging
import pickle
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
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from features import gen_msg_features, dont_imperative, keywords_general
from lib import NLTKPreprocessor, analysis, feature_importances
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn import svm
import multiprocessing as mp
import matplotlib.pyplot as plt
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cat_map = {1: 'Emergency', 2: 'To-Do', 3: 'General'}
sub_cat_map = {11: 'E_Fire', 12: 'E_Health', 13: 'E_Personal', 14: 'E_Police',\
                21: 'TD_Event', 22: 'TD_Non_Event', 23: 'TD_Urgent_Callback',
               24: 'TD_Urgent_NonCallBack', 31: 'G_General'}
clf_map = {
    1: [LogisticRegression(penalty='l2', C=10), {'C':[1,10,100], 'penalty':
                                                 ['l1', 'l2']}],
            2: [RandomForestClassifier(n_estimators = 100, max_features='sqrt'),{}],
            3: [svm.LinearSVC(), {}],
            4: [GradientBoostingClassifier(), {}]
        }
clf_map[5] = VotingClassifier(estimators=[('0', clf_map[1]), ('1', clf_map[2]),\
                                         ('2', clf_map[3])])

def roc_auc(X_test, y_test, cls):
    y_preds = cls.decision_function(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_preds.shape[1]
    y = label_binarize(y_test, classes=[0,1,2])
    #pdb.set_trace()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], y_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    #plt.plot(fpr["micro"], tpr["micro"],
    #        label='micro-average ROC curve (area = {0:0.2f})'
    #            ''.format(roc_auc["micro"]))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

def process_data(train, test, val, args):
    filter_col = 'Category'
    exclude_list_str = []
    if args.level == 2:
        exclude_list_str = [sub_cat_map[args.level*10+i] for i in args.exclude]
        filter_list = [cat_map[args.top_level]]
        # Extract Data of a single top-level category
        train = train[train["Category"].isin(filter_list)]
        test = test[test["Category"].isin(filter_list)]
        val = val[val["Category"].isin(filter_list)]
        
        # Exclude sub-categories if specified
        train = train[~train["Sub-Category"].isin(exclude_list_str)]
        test = test[~test["Sub-Category"].isin(exclude_list_str)]
        val = val[~val["Sub-Category"].isin(exclude_list_str)]
        filter_col = 'Sub-Category'
    else:
        exclude_list_str = [cat_map[i] for i in args.exclude]
        # exclude top-level categories if specified
        train = train[~train["Category"].isin(exclude_list_str)]
        test = test[~test["Category"].isin(exclude_list_str)]
        val = val[~val["Category"].isin(exclude_list_str)]
    
    X_train = train
    y_train = train[filter_col]
    X_test = test
    y_test = test[filter_col]
    X_val = val
    y_val = val[filter_col]
        
    build_and_evaluate(X_train, y_train, X_test, y_test, X_val, y_val , args,
        clf_map[args.clf][0], clf_map[args.clf][1])


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
    cmap = plt.get_cmap('viridis')
    index = np.arange(len(encoder.classes_))
    colors = ['r', 'b', 'g', 'y', 'c', 'm', '#c21211', '#000000', '#00aa11',
              '#22cc22', '#bfe23e']
    colors = cmap(np.linspace(0, 1, len(labels)))
    #labels = np.append(labels,'total')
    for idx1, id1 in enumerate(labels):
        plt.bar(index+0.05*idx1, values[idx1], width=0.02,color=colors[idx1],
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


def generate_labels(y_train, y_test, y_val):
    # Label encode the targets
    labels = LabelEncoder()
    y_train = labels.fit_transform(y_train)
    y_test = labels.transform(y_test)
    y_val = labels.transform(y_val)
    print(labels.classes_)
    return labels, y_train, y_test, y_val


def misclassifications_class(model, Xte, yte, msgs, label_enc, level, wtf=False):
    y_preds = model.predict(Xte)
    misc = np.where(y_preds != yte)
    # select all misclassified 
    df = msgs.iloc[misc]
    misc_labels = label_enc.inverse_transform(y_preds)[misc]
    index = 'Category'
    if level == 2:
        index = 'Sub-Category'
    for sc in label_enc.classes_:
        total = msgs[msgs[index] == sc].shape[0]
        miss = df[df[index] == sc].shape[0]
        print("%s Miss: %s/%s(%04.2f)%%"%(sc, miss, total, (miss/max(total,1))*100))
        if wtf:
            miss_data = df[df[index] == sc]
            miss_data['Classifier Label'] = misc_labels[df[index] == sc]
            miss_data.to_csv("misclassified/"+sc.lower()+'.tsv', sep = '\t')


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


def sk_to_weka(X, y, header, filename = 'weka.arff'):
    f = open(filename, 'w')
    f.write('@RELATION SMS\n\n')
    for h in header[:-1]:
        f.write('@ATTRIBUTE {} REAL\n'.format(h))
    f.write('@ATTRIBUTE class {{{}}}\n\n@DATA\n'.format(','.join(header[-1])))
    for idx, x in enumerate(X):
        line = ','.join(map(lambda s: str(s), x)) + ',' + str(header[-1][y[idx]]) + '\n'
        f.write(line)
    f.close()


def generate_features(transformer, X, type=''):
    vectors = np.array(gen_msg_features(X, type))
    X_mat = transformer.transform(X)
    X_mat = sp.sparse.hstack((X_mat, vectors[1]), format='csr')
    return [X_mat, vectors[0]]
    #return [vectors[1], vectors[0]]


def cross_validate(model, X, y, args):
    logger.info("Performing {} fold cross validation".format(args.cv))
    kf = KFold(n_splits=args.cv, shuffle=True)
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        print(model.score(X[test], y[test]))


def load_and_test(X_test, y_test, args):
    labels, y_test, _, _ = generate_labels(y_test, [], [])

    model = pickle.load(open(args.model, 'rb'))

    X_test = readDocuments(args.test, encoding = None, skip_header = True)

    X_test_mat, f_test_labels = generate_features(model['vect'], X_test["Message"])
    y_preds = model['cls'].predict(X_test_mat)
    t_lb = labels.inverse_transform(y_preds)
    pdb.set_trace()
    X_test["Category-Predicted"] = t_lb
    X_test.to_csv(args.test+'.labels', sep='\t')
    return y_preds


def check_msg(x):
	return (dont_imperative(x) or keywords_general(x))

def postProcess(y_pred,test_msgs,label_enc):
    test_msgs = list(map(lambda a:a.lower().strip('., '), test_msgs))
    test_msgs = list(map(lambda a:re.sub('[\.]', ' . ',a), test_msgs))
    test_msgs = list(map(lambda a:re.sub('[,]', ' , ',a), test_msgs))
    y_labels = label_enc.inverse_transform(y_pred)
	
    modified_y = []
    for y, x in zip(y_labels,test_msgs):
        if((y in ['emergency','urgent']) and check_msg(x)):
            modified_y.append('general')
        else:
            modified_y.append(y)

    return label_enc.transform(modified_y)


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
     
    labels, y_train, y_test, y_val = generate_labels(y_train, y_test, y_val)
    #labels = LabelEncoder()
    #try:
    #    y_test = labels.fit_transform(y_test)
    #    y_val = labels.fit_transform(y_val)
    #except:
    #    type, value, tb = sys.exc_info()
    #    traceback.print_exc()
    #    pdb.post_mortem(tb)
    
    Xtr, Xte, ytr, yte = X_train, X_test, y_train, y_test
    
    prep = NLTKPreprocessor(stem=True)
    vect = TfidfVectorizer(preprocessor=prep,
                lowercase=True, stop_words=None, ngram_range=(1,2))
    vect.fit_transform(Xtr["Message"])

    X_tr_mat, f_tr_labels = generate_features(vect, Xtr["Message"], args.data+'train')
    X_te_mat, f_te_labels = generate_features(vect, Xte["Message"], args.data+'tune')
    X_val_mat, f_val_labels = generate_features(vect, X_val["Message"], args.data+'test')
    # write train to weka for feature analysis
    if args.to_weka:
        vectors_train = np.array(gen_msg_features(Xtr["Message"],args.data+'train'))
        vectors_test = np.array(gen_msg_features(Xte["Message"], args.data+'tune' ))
        vectors_val = np.array(gen_msg_features(X_val["Message"], args.data+'test'))
        
        header_train = f_tr_labels.tolist()
        header_train.append(labels.classes_)
        sk_to_weka(vectors_train[1], ytr, header_train,\
                   '_'.join(labels.classes_)+'_train_features.arff')
        sk_to_weka(vectors_test[1], y_test, header_train,\
                   '_'.join(labels.classes_)+'_tune_features.arff')
        sk_to_weka(vectors_val[1], y_val, header_train,\
                   '_'.join(labels.classes_)+'_test_features.arff')
    
    cls = classifier
    
    if args.gridsearch:
        logger.info("Performing GridSearch on train data")
        clf = GridSearchCV(estimator=cls, param_grid=parameters)
        clf.fit(X_tr_mat, ytr)
        best_parameters = clf.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    if args.clf >= 4:
        X_tr_mat = X_tr_mat.toarray()
        X_te_mat = X_te_mat.toarray()
        X_val_mat = X_val_mat.toarray()
    if args.cv != 0:
        cross_validate(cls, X_tr_mat, ytr, args)
    cls.fit(X_tr_mat, ytr)
    
    y_pred = (cls.predict(X_val_mat))

    y_pred = postProcess(y_pred,X_val["Message"],labels)

    print(confusion_matrix(y_val, y_pred))
    conf_f_name = '_'.join(labels.classes_) + '_cfm.tsv'
    
    np.savetxt(conf_f_name, confusion_matrix(y_val, y_pred), delimiter='\t',\
               fmt="%2.1d", header='\t'.join(labels.classes_))
    scores = clsr(y_val, y_pred)
    scores = list(map(lambda r: re.sub('\s\s+', '\t', r),\
                                scores.split("\n")))
    #scores[0] = '\t' + scores[0]
    scores[-2] = '\t' + scores[-2]
    scores = '\n'.join(scores)

    print(scores)
    with open('_'.join(labels.classes_) + '_scores.tsv', 'w') as f:
        f.write('\t'+scores)
    
    if args.save:
        fitted_model = {'vect': vect, 'cls': cls}
        logger.info("Saving model")
        pickle.dump(fitted_model,\
                    open('_'.join(sorted(labels.classes_))+'_'+str(cls)[0:10]+'.model', 'wb'))
    misclassifications_class(cls, X_val_mat, y_val, X_val,
                            labels, args.level, True)
    if args.with_graph:
        plot_feat(vectors_val[1], vectors_val[0], labels, y_val,
                    'Average Feature value for each class')
    if args.roc:
        roc_auc(X_val_mat, y_val, cls)


def readDocuments(filename, text_col=0, tag_col=1, skip_header=False, encoding='utf-8'):
    documents = []
    labels = []

    with open(filename, 'r', newline='', encoding=encoding) as data_file:
        csvfile = csv.reader(data_file, delimiter="\t")

        for line in csvfile:
            if skip_header==True:
                skip_header=False
                continue
            documents.append(line[text_col].strip())
            labels.append(line[tag_col].strip().lower())

    data = {'Message': documents, 'Category': labels}
    return pd.DataFrame(data)

def readVal(filename, text_col=0, tag_col=1, additional_col=2, skip_header=False, encoding='utf-8'):
    documents = []
    labels = []
    additional = []
    with open(filename, 'r', newline='', encoding=encoding) as data_file:
        csvfile = csv.reader(data_file, delimiter="\t")

        for line in csvfile:
            if skip_header==True:
                skip_header=False
                continue
            documents.append(line[text_col].strip())
            labels.append(line[tag_col].strip().lower())
            additional.append(line[additional_col].strip().lower())

    data = {'Message': documents, 'Category': labels, 'WSD/PT/Neg' : additional}
    return pd.DataFrame(data)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Classify SMS messages')

    parser.add_argument('--data', type=str,
                        help="Data file prefix")
    parser.add_argument('--level', type=int, choices=(1, 2), default=1,
                        help="Level to classify on")
    parser.add_argument('--top_level', type=int, choices=(1, 2, 3), default=1,
                        help="Top level category to filter if level 2,\
                        Emergency, To-Do, General")
    parser.add_argument('--exclude', type=int, nargs="+",default=[],\
                        help="Categories or Sub-Categories to exclude")
    parser.add_argument('--clf', type=int, choices=(1, 2, 3, 4, 5, 6, 7), default=1,
                        help="Classifier: Logistic Regression, Random Forest,\
                        GBC")
    parser.add_argument('--with_graph', action='store_true',
                        help="Whether to print feature graph")
    parser.add_argument('--to_weka', action='store_true',
                        help="Whether to print feature graph")
    parser.add_argument('--save', action='store_true',
                        help="Whether to save the model")
    parser.add_argument('--model', type=str,
                        help="Model filename, use with --predict")
    parser.add_argument('--test', type=str,
                        help="test filename, use with --predict")
    parser.add_argument('--predict', action='store_true',
                        help="Whether to generate preds from stored model")
    parser.add_argument('--gridsearch', action='store_true',
                        help="GridSearch on classifier parameters")
    parser.add_argument('--roc', action='store_true',
                        help="Whether to generate preds from stored model")
    parser.add_argument('--cv', type=int, default=0,
                        help="cross validation, default 0")

    parser.parse_args()
    args = parser.parse_args()
     
    if args.predict:
        X_test = readDocuments(args.test, encoding = None, skip_header = True)
        y_test = X_test['Category']
        y_preds = load_and_test(X_test, y_test, args)
    else:
        trainF =args.data + "train.tsv"
        tuneF = args.data + "tune.tsv"    
        testF = args.data + "test.tsv"
        
        train = readDocuments(trainF, encoding = None, skip_header=True)
        test = readDocuments(tuneF, encoding = None, skip_header=True)
        val = readVal(testF, encoding = None, skip_header=True)
        
        process_data(train, test, val, args)
