""" pretrain a word2vec on the corpus"""
import argparse
import json
import logging
import os
from os.path import join, exists
import pandas as pd
from time import time
from datetime import timedelta

import gensim

class Sentences(object):
    """ needed for gensim word2vec training"""
    def __init__(self, args):
        self._path = args.data_dir

    def __iter__(self):
        df = pd.concat([pd.read_csv(join(self._path, 'train.csv'), lineterminator='\n'), pd.read_csv(join(self._path, 'val.csv'), lineterminator='\n')])
        df = df.dropna().reset_index(drop=True)
        for i, row in df.iterrows():
            if row['sentence']:
                yield row['sentence'].split(' ')

def main(args):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    start = time()
    save_dir = args.path
    if not exists(save_dir):
        os.makedirs(save_dir)

    sentences = Sentences(args)
    model = gensim.models.Word2Vec(
        size=args.dim, min_count=5, workers=16, sg=1)
    model.build_vocab(sentences)
    print('vocab built in {}'.format(timedelta(seconds=time()-start)))
    model.train(sentences,
                total_examples=model.corpus_count, epochs=model.iter)

    model.save(join(save_dir, 'word2vec.{}d.{}k.bin'.format(
        args.dim, len(model.wv.vocab)//1000)))
    model.wv.save_word2vec_format(join(
        save_dir,
        'word2vec.{}d.{}k.w2v'.format(args.dim, len(model.wv.vocab)//1000)
    ))

    print('word2vec trained in {}'.format(timedelta(seconds=time()-start)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train word2vec embedding used for model initialization'
    )
    parser.add_argument('--path', required=True, help='path to store word embeddings')
    parser.add_argument('--data_dir', type=str, required=True, help='Location of the data corpus')
    parser.add_argument('--dim', action='store', type=int, default=100)
    args = parser.parse_args()

    main(args)
