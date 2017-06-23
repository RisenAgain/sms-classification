# -*- coding: utf-8 -*-
from sklearn.preprocessing import RobustScaler, StandardScaler
import pandas as pd
import numpy as np
import pdb
import re
import os
modals = set(['can', 'could', 'may', 'might', 'will', 'would', 'shall',
              'should', 'must'])
questions = set(['who', 'what', 'where', 'when', 'which', 'how', 'why'])
temporal = set(['meet','tonight', 'today', 'tomorrow', 'min', 'month', 'day',
                'afternoon', 'morning', 'sunday', 'monday', 'tuesday', 
               'wednesday','thursday', 'friday',
                'saturday','january','february','march',
                'april','may','june','july','august','september','october','november','december'])


def personal(x):
    regex = 'my|ours|us|we|i|me'
    if re.search(regex, x):
        return 1
    return 0


def fire(x):
    regex = 'fire|smoke|flame|burn|caught|evacu|explo| blaze| kill | injure | destroy| ignite '
    z = re.findall(regex, x)
    if z:
        return len(z)
    return 0


def health(x):
    regex =\
    'hospit|ambul|medic|disease|headache|doctor|bleed|blood|pain|heart|critical|injure'
    z=re.findall(regex, x) 
    if z:
        return len(z)
    return 0


def emer(x):
    return fire(x)+health(x)


def todo(x):
    return meet_suggest(x)+date(x)


def request_immedi(x):
    regex = 'pls| please | now | asap | min |immediate'
    if re.search(regex, x):
        return 1
    return 0


def modal_verbs(x):
    words = set(x.split())
    if len(words.intersection(modals)) > 0:
        return 1
    return 0


def meet_suggest(x):
    if re.search('what |schedule| should |shall| once |may be| be \
                 available | meet | can | let', x):
        return 1
    return 0


def date(x):
    words = set(x.split())
    if len(words.intersection(temporal)) > 0:
        return 1
    if\
    re.search('[0-9]-[0-9]|morning|afternoon|evening|midnight|month|day|year|week|[0-9]\.[0-9]|[0-9]\s*[ap]m|[0-9]\s*[ap].m|[0-9]\s*min|[0-9]\s*hours|[0-9]:[0-9]|[0-9]\s*today|tomorrow|0th|[4-9]th|1st|2nd|3rd|[0-9]/[0-9]|o\'clock', x):
        return 1
    return 0
def msg_len_word(x):
    return len(x)


def puncts(x):
    z = re.findall('[:,\.\/-]|\+', x)
    if z:
        return len(z)
    return 0


def msg_len_char(x):
    return len(x.split())


def capitall(x):
    capitals = 0
    for l in x:
        if l.isupper():
            capitals += 1
    return capitals


def capitaln(x):
    capitals = 0
    for l in x:
        if l.isupper():
            capitals += 1
    return capitals / max(len(x.split()),1)


def question(x):
    words = set(x.split())
    if len(words.intersection(questions)) > 0:
        return 1
    if re.search('?',x):
        return 1
    return 0


def call(x):
    regex = 'call|immediate|bring|asap|reply'
    z = re.findall(regex,x)
    if z:
        return len(z)
    return 0


def numeric(x):
    z = re.findall('[0-9]+', x)
    if z:
        return len(z)
    return 0


def gen_msg_features(X):
    X = list(map(lambda a:a.lower().strip('., '), X))
    X = list(map(lambda a:re.sub('[\.]', ' . ',a), X))
    X = list(map(lambda a:re.sub('[\']', ' ',a), X))
    X = list(map(lambda a:re.sub('[,]', ' , ',a), X))
    #X = list(map(lambda a:re.sub('\s+', '\s',a), X))
    feature_names = [request_immedi, puncts, msg_len_char, msg_len_word, date,
                     call, numeric, meet_suggest, health, fire  ]
    names = np.array(list(map(lambda a: a.__qualname__, feature_names)))
    feature_set = np.empty((0,len(names)))
    for x in X:
        features = np.array(list(map(lambda f: f(x), feature_names)))
        feature_set = np.append(feature_set,[features], axis=0)
    scaler = StandardScaler(with_mean=False)
    feature_set = scaler.fit_transform(feature_set)
    return [names, feature_set]

