# -*- coding: utf-8 -*-
from sklearn.preprocessing import RobustScaler, StandardScaler
from nltk.parse.stanford import StanfordParser
from nltk import tree
import nltk
import pandas as pd
import logging
import numpy as np
import pdb
import re
from nltk.wsd import lesk
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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_words(filename):
    words = []
    try:
        with open(filename, 'r') as fn:
            words = fn.read().split('\n')
    except EnvironmentError as e:
        logger.warn("Failed to read file {}".format(filename))
    return filter(None, words)


fire_regex = '|'.join(get_words('fire.words'))
health_regex = '|'.join(get_words('health.words'))
personal_regex = '|'.join(get_words('personal.words'))
wsdf = pd.read_csv('wsd_features', header=0, delimiter='\t')
wsd_words = wsdf['Word'].tolist()

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


def personal(x,dependency_tree, dependency_relations):
    regex = personal_regex
    if re.search(regex, x):
        return 1
    return 0


def fire(x,dependency_tree, dependency_relations):
    regex = fire_regex
    foundMatch = 0
    matchw = None

    if(dont_imperative(x)):
        return 0
    if(keywords_general(x)):
        return 0
    return check_negation(regex, x, dependency_tree, dependency_relations)

    if foundMatch and matchw in wsd_words:
        if validate_wsd(matchw, x):
            return 1
        else:
            return 0
    return foundMatch


def health(x,dependency_tree, dependency_relations):
    regex = health_regex
    foundMatch = 0
    matchw = None
    if(dont_imperative(x)):
        return 0
    if(keywords_general(x)):
        return 0
    return check_negation(regex, x, dependency_tree, dependency_relations)
    if foundMatch and matchw in wsd_words:
        if validate_wsd(matchw, x):
            return 1
        else:
            return 0
    return foundMatch


def emer(x, dependency_tree, dependency_relations):
    return fire(x,dependency_tree, dependency_relations)+health(x,dependency_tree, dependency_relations)


def todo(x, dependency_tree, dependency_relations):
    return meet_suggest(x, dependency_tree, dependency_relations)+date(x, dependency_tree, dependency_relations)


def request_immedi(x, dependency_tree, dependency_relations):
    regex = 'pls| please | now | asap | min |immediate'
    foundMatch = 0
    if(dont_imperative(x)):
        return 0
    if(keywords_general(x)):
        return 0
    return check_negation(regex, x, dependency_tree, dependency_relations)
    return foundMatch


def modal_verbs(x):
    words = set(x.split())
    if len(words.intersection(modals)) > 0:
        return 1
    return 0


def meet_suggest(x, dependency_tree, dependency_relations):
    if(dont_imperative(x)):
        return 0
    if(keywords_general(x)):
        return 0
    regex = 'what |schedule| should |shall| once |may be| be \
                 available |meet| can | let'
    return check_negation(regex, x, dependency_tree, dependency_relations)


def date(x, dependency_tree, dependency_relations):
    words = set(x.split())
    match = list(words.intersection(temporal))
    if len(match) > 0:
        return 1
    #if lev_match(temporal, words):
    #    return 1
    regex = '[0-9]-[0-9]|morning|afternoon|evening|midnight|month|day|year|week|[0-9]\.[0-9]|[0-9]\s*[ap]m|[0-9]\s*[ap].m|[0-9]\s*min|[0-9]\s*hours|[0-9]:[0-9]|[0-9]\s*today|tomorrow|0th|[4-9]th|1st|2nd|3rd|[0-9]/[0-9]|o\'clock'
    return check_negation(regex, x, dependency_tree, dependency_relations)


def msg_len_word(x, dependency_tree, dependency_relations):
    return len(x)


def puncts(x, dependency_tree, dependency_relations):
    z = re.findall('[:,\.\/#=\?:0-9\~<>!\-]|\+', x)
    if z:
        return len(z)
    return 0


def msg_len_char(x, dependency_tree, dependency_relations):
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


def call(x, dependency_tree, dependency_relations):
    x = x.lower()
    if(dont_imperative(x)):
        return 0
    if(keywords_general(x)):
        return 0
    regex = 'call|immediate|bring|asap|reply'
    #z = re.findall(regex,x)
    foundMatch = 0
    return check_negation(regex, x, dependency_tree, dependency_relations)
    return foundMatch


def numeric(x, dependency_tree, dependency_relations):
    z = re.findall('[0-9]+|X+', x)
    if z:
        return len(z)
    return 0


def check_negation(regex, x, dependency_tree, dependency_relations):
    foundMatch = 0
    if(len(dependency_relations) != 0 and len(dependency_relations[0]) != 0):
        for match in finditer(regex, x):
            foundMatch = 1
            if(traverse_dep_tree(match.group(),dependency_tree,dependency_relations)):
                return 0
    return foundMatch


def traverse_dep_tree(keyword,tree,relations):
    siblings = []
    parent = None
    stack = [[tree,parent,siblings]]
    neg = False

    #iterative DFS
    while(stack != []):
        root = stack.pop()
        label = None
        if(type(root[0]) == nltk.Tree):
            label = root[0].label()
            # print(label)
            if(label == keyword):
                neg |= check_relation(root[1],root[2],relations)
                neg |= check_relation(label,root[0][:],relations)
        else:
            label = root[0]
            # print(label)
            if(label == keyword):
                neg |= check_relation(root[1],root[2],relations)

        if(type(root[0]) == nltk.Tree):
            for i in reversed(root[0]):
                stack.append([i,root[0].label(),root[0][:]])

    return neg


def check_relation(parent,children,relations):
    children = [child.label() if(type(child) == nltk.Tree) else child for child in children]
    relations = relations[0]
    # print(parent,children)
    for relation in relations:
        for child in children:
            if(parent == relation[0][0] and child == relation[2][0] and relation[1] == 'neg'):
                return True
    return False

def dont_imperative(x):
    if(len(x) != 0):
        first_word = x.strip().split()[0].lower()
        if(first_word == "don't" or first_word == "dont"):
            return True
    return False

#keywords strongly suggesting a general sentence
def keywords_general(x):
    keywords = ["never","ever"]
    x = x.lower().split()
    if any(key in x for key in keywords):
        return True
    else:
        return False


def tense(x,dependency_tree, dependency_relations):
    # 0 - past, 1 - present, 2 - future, 3 - unkown
    sent_tense = 3
    if(len(dependency_relations) == 0 or len(dependency_relations[0]) == 0):
        return sent_tense
    found_verb = None
    present_verbs = ['VB', 'VBG', 'VBP', 'VBZ']
    past_verbs = ['VBD', 'VBN']
    future_verbs = ['MD']
    for relation in dependency_relations[0]:
        for verb in past_verbs:
            if(verb in relation[0][1] or verb in relation[2][1]):
                sent_tense = 0
        if(sent_tense != 3):
            break
        for verb in future_verbs:
            if(verb in relation[0][1] or verb in relation[2][1]):
                sent_tense = 2
        if(sent_tense != 3):
            break
        for verb in present_verbs:
            if(verb in relation[0][1]):
                sent_tense = 1
                found_verb = relation[0][0]
            elif(verb in relation[2][1]):
                sent_tense = 1
                found_verb = relation[2][0]
        if(sent_tense != 3):
            break
    if(sent_tense == 1):
        for relation in dependency_relations[0]:
            if(relation[0][0] == found_verb):
                if(relation[2][1] in past_verbs):
                    sent_tense = 0
                elif(relation[2][1] in future_verbs):
                    sent_tense = 2
    return (sent_tense/3.0)


def validate_wsd(word, sent):
    sense = lesk(sent.split(), word).name()
    sets = wsdf[wsdf['Word']==word]['Synset'].values[0] 
    return sense in sets.split(',')


def pos_feat(names, feature_set, dataf):
    pos_file = open(dataf+'_pos', 'r')
    sents = [l.strip() for l in pos_file]
    present_verbs = ['VB', 'VBG', 'VBP', 'VBZ']
    past_verbs = ['VBD', 'VBN']
    verb_feats = np.empty((0,2))
    for s in sents:
        tokens = s.split()
        tag_list = []
        for t in tokens:
            try:
                tag = t.split('_')[1]
                tag_list.append(tag)
            except:
                pass
        feats = []
        for pv in present_verbs:
            if pv in tag_list and len(feats) == 0:
                feats.append(1)
        if len(feats) == 0:
            feats.append(0)

        for pv in past_verbs:
            if pv in tag_list and len(feats) == 1:
                feats.append(1)
        if len(feats) == 1:
            feats.append(0)
        
        verb_feats = np.append(verb_feats, [feats], axis=0)
    names = np.append(names, ['present', 'past'])
    try:
        feature_set = np.hstack((feature_set, verb_feats))
    except:
        pdb.set_trace()
    return names, feature_set


def gen_feat_arr(X, feature_names, dependency_tree, dependency_relations):
    feature_set = np.empty((0,len(feature_names)))
    print(len(X),len(dependency_tree),len(dependency_relations))
    for i in range(len(X)):
        features = np.array(list(map(lambda f: f(X[i],dependency_tree[i],dependency_relations[i]), feature_names)))
        feature_set = np.append(feature_set,[features], axis=0)
    return feature_set


def handle_chunks(dataf):
    chunk_file = open(dataf+'_chunk', 'r')
    sents = [l.strip() for l in chunk_file]
    chunk_f = list(map(lambda f: do_chunk(f), sents[1:-1]))
    return np.array(chunk_f)


def gen_msg_features(X, dependency_tree, dependency_relations, dataf = '', procs = 1):
    X = list(map(lambda a:a.lower().strip('., '), X))
    X = list(map(lambda a:re.sub('[\.]', ' . ',a), X))
    #X = list(map(lambda a:re.sub('[\']', ' ',a), X))
    X = list(map(lambda a:re.sub('[,]', ' , ',a), X))
    #X = list(map(lambda a:re.sub('\s+', '\s',a), X))
    feature_names = [request_immedi, puncts, msg_len_char, msg_len_word,
                     call, numeric,tense]
    top_level = [emer, todo]
    second_level = [date, meet_suggest]
    feature_names += top_level

    names = np.array(list(map(lambda a: a.__qualname__, feature_names)))
    
    feature_set = gen_feat_arr(X, feature_names, dependency_tree, dependency_relations)
    #names, feature_set = pos_feat(names, feature_set, dataf)
    
    # handle Chunking
    #names = np.append(names, 'chunk_NP')
    #names = np.append(names, 'chunk_VP')
    #chunk_f = handle_chunks(dataf)
    #feature_set = np.append(feature_set, chunk_f[:,0].reshape((feature_set.shape[0],1)), axis=1)
    #feature_set = np.append(feature_set, chunk_f[:,1].reshape((feature_set.shape[0],1)), axis=1)

    scaler = StandardScaler(with_mean=False)
    feature_set = scaler.fit_transform(feature_set)
    return [names, feature_set]

