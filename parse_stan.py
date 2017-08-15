from nltk.parse.stanford import StanfordParser
import multiprocessing as mp
import argparse
import re
import pdb
LIB = 'libs/stanford-parser-full-2016-10-31/'
#english_parser = StanfordParser(LIB+'stanford-parser.jar',\
#                                LIB+'stanford-parser-3.7.0-models.jar')
sents = ['please meet me outside the office']
#res = english_parser.parse_one([sents[0]])

def parse_sents(sents):
    parser = StanfordParser(LIB+'stanford-parser.jar',\
                                LIB+'stanford-parser-3.7.0-models.jar')
    res = []
    for s in sents:
        try:
            tree = str(parser.parse_one([s])).replace('\n', '')
            tree = re.sub('\s+', ' ', tree)
            res.append(tree)
        except:
            print("Exception occures:{}".format(s))
            res.append('')
    return res

def write_trees(sents, outfile):
    fo = open(outfile, 'w')
    for s in sents:
        fo.write(s)
        fo.write('\n')
    fo.close()

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Classify SMS messages')

    parser.add_argument('--data', type=str,
                        help="Data file path")
    parser.add_argument('--file', type=str,
                        help="file")
    parser.add_argument('--cores', type=int,
                        help="num cores")
    cores = 1
    parser.parse_args()
    args = parser.parse_args()
    if args.cores:
        cores = args.cores
    fin = open(args.data+args.file, 'r')
    sents = fin.read().split('\n')
    pool = mp.Pool(processes=cores)
    res = pool.apply(parse_sents, args=(sents,))
    write_trees(res, args.data+args.file+'_stantrees')
