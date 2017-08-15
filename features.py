# -*- coding: utf-8 -*-
from sklearn.preprocessing import RobustScaler, StandardScaler
from nltk.parse.stanford import StanfordParser
from nltk import tree
import pandas as pd
import numpy as np
import pdb
import re
from re import finditer
import os
import editdistance
import multiprocessing as mp
HALF_WINDOW = 28

modals = set(['can', 'could', 'may', 'might', 'will', 'would', 'shall',
              'should', 'must'])
questions = set(['who', 'what', 'where', 'when', 'which', 'how', 'why'])
temporal = set(['tonight', 'today', 'tomorrow', 'min', 'month', 'day',
                'afternoon', 'morning', 'sunday', 'monday', 'tuesday', 
               'wednesday','thursday', 'friday',
                'saturday','january','february','march',
                'april','may','june','july','august','september','october','november','december'])

#st_parser = StanfordParser('libs/stanford-parser-full-2016-10-31/stanford-parser.jar',
#                        'libs/stanford-parser-full-2016-10-31/stanford-parser-3.7.0-models.jar')


def chunk_helper(chunk):
    chunks = []
    chunk_tags = set()
    start = 0
    try:
        while True:
            curr_i = chunk.index('[', start)
            curr_chunk = chunk[curr_i+1:chunk.index(']', start)]
            curr_chunk_split = curr_chunk.split(' ')
            chunks.append((curr_chunk_split[0], ' '.join(curr_chunk_split[1:])))
            chunk_tags.add(curr_chunk_split[0])
            start = chunk.index(']', curr_i) + 1
    except:
        pass
    return chunks, chunk_tags


def do_chunk(x):
    splitted = x.split('[.,\&]')
    for split in splitted:
        chunks, tags = chunk_helper(split)
        pdb.set_trace()
        for tag in tags:
            if tag == 'NP':
                break
            # VP found before NP
            if tag == 'VP':
                return 1
    return 0


def traverse_tree(deps, level, nodes = {}):
    if level > 4:
        return 0
    for subtree in deps:
        if type(subtree) == tree.Tree and subtree.label() == 'S':
            ref = 0
            for child in subtree:
                if child.label() == 'NP':
                    ref = ref | 2
                if child.label() == 'VP':
                    ref = ref | 1
            return ref
        else:
            return 0
        if type(subtree) == tree.Tree:
            return traverse_tree(subtree, level + 1)


def parser(x):
    deps = st_parser.parse_one([x])
    status = traverse_tree(deps, 0)
    val = 1 if status == 1 else 0
    return val


def lev_match(ref_list, msg_words):
    for w in msg_words:
        for r in ref_list:
            if editdistance.eval(w, r) <= 2:
                return 1


def personal(x):
    regex = 'my|ours|us|we|i|me'
    if re.search(regex, x):
        return 1
    return 0


def fire(x):
    regex = 'fire|smoke|flame|burn|caught|evacu|explo|blaze|kill|injure|destroy|ignite '
    foundMatch = 0
    for match in finditer(regex, x):
        foundMatch = 1
        left = max(match.span()[0] - HALF_WINDOW, 0)
        right = min(match.span()[1]+HALF_WINDOW, len(x))
        if 'not' in x[left:right] or 'don\'t' in x[left:right]:
            return 0
    return foundMatch


def health(x):
    regex =\
    'hospit|ambul|medic|disease|headache|doctor|bleed|blood|pain|heart|critical|injure'
    foundMatch = 0
    for match in finditer(regex, x):
        foundMatch = 1
        left = max(match.span()[0] - HALF_WINDOW, 0)
        right = min(match.span()[1]+HALF_WINDOW, len(x))
        if 'not' in x[left:right] or 'don\'t' in x[left:right]:
            return 0
    return foundMatch


def emer(x):
    return fire(x)+health(x)


def todo(x):
    return meet_suggest(x)+date(x)


def request_immedi(x):
    regex = 'pls| please | now | asap | min |immediate'
    foundMatch = 0
    for match in finditer(regex, x):
        foundMatch = 1
        left = max(match.span()[0] - HALF_WINDOW, 0)
        right = min(match.span()[1]+HALF_WINDOW, len(x))
        if 'not' in x[left:right] or 'don\'t' in x[left:right]:
            return 0
    return foundMatch


def modal_verbs(x):
    words = set(x.split())
    if len(words.intersection(modals)) > 0:
        return 1
    return 0


def meet_suggest(x):
    regex = 'what |schedule| should |shall| once |may be| be \
                 available |meet| can | let'
    return match_regex(regex, x)


def date(x):
    words = set(x.split())
    match = list(words.intersection(temporal))
    if len(match) > 0:
        return 1
    #if lev_match(temporal, words):
    #    return 1
    regex = '[0-9]-[0-9]|morning|afternoon|evening|midnight|month|day|year|week|[0-9]\.[0-9]|[0-9]\s*[ap]m|[0-9]\s*[ap].m|[0-9]\s*min|[0-9]\s*hours|[0-9]:[0-9]|[0-9]\s*today|tomorrow|0th|[4-9]th|1st|2nd|3rd|[0-9]/[0-9]|o\'clock'
    return match_regex(regex, x)


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
    x = x.lower()
    regex = 'call|immediate|bring|asap|reply'
    #z = re.findall(regex,x)
    foundMatch = 0
    for match in finditer(regex, x):
        foundMatch = 1
        left = max(match.span()[0] - HALF_WINDOW, 0)
        right = min(match.span()[1]+HALF_WINDOW, len(x))
        if 'not' in x[left:right] or "don't" in x[left:right]:
            return 0
    return foundMatch


def numeric(x):
    z = re.findall('[0-9]+|X+', x)
    if z:
        return len(z)
    return 0


def match_regex(regex, x):
    foundMatch = 0
    for match in finditer(regex, x):
        foundMatch = 1
        left = max(match.span()[0] - HALF_WINDOW, 0)
        right = min(match.span()[1]+HALF_WINDOW, len(x))
        if 'not' in x[left:right] or "don't" in x[left:right]:
            return 0
    return foundMatch


def gen_feat_arr(X, feature_names):
    feature_set = np.empty((0,len(feature_names)))
    for x in X:
        try:
            features = np.array(list(map(lambda f: f(x), feature_names)))
            feature_set = np.append(feature_set,[features], axis=0)
        except:
            pdb.set_trace()
    return feature_set


def handle_chunks(dataf):
    chunk_file = open(dataf+'_chunk', 'r')
    sents = [l.strip() for l in chunk_file]
    chunk_f = list(map(lambda f: do_chunk(f), sents[1:-1]))
    return np.array(chunk_f)


def gen_msg_features(X, dataf = '' , procs = 1):
    X = list(map(lambda a:a.lower().strip('., '), X))
    X = list(map(lambda a:re.sub('[\.]', ' . ',a), X))
    #X = list(map(lambda a:re.sub('[\']', ' ',a), X))
    X = list(map(lambda a:re.sub('[,]', ' , ',a), X))
    #X = list(map(lambda a:re.sub('\s+', '\s',a), X))
    feature_names = [request_immedi, puncts, msg_len_char, msg_len_word,
                     call, numeric]
    top_level = [emer, todo]
    second_level = [date, meet_suggest]
    feature_names += top_level

    names = np.array(list(map(lambda a: a.__qualname__, feature_names)))
    #pool = mp.Pool(processes = procs)
    #feature_set = pool.map(gen_feat_arr, X)
    feature_set = gen_feat_arr(X, feature_names)
    # handle Chunking
    #names = np.append(names, 'chunk_NP')
    #names = np.append(names, 'chunk_VP')
    #chunk_f = handle_chunks(dataf)
    #feature_set = np.append(feature_set, chunk_f[:,0].reshape((feature_set.shape[0],1)), axis=1)
    #feature_set = np.append(feature_set, chunk_f[:,1].reshape((feature_set.shape[0],1)), axis=1)

    scaler = StandardScaler(with_mean=False)
    feature_set = scaler.fit_transform(feature_set)
    return [names, feature_set]

