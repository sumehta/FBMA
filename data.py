import os
import torch
import codecs
import gensim
import logging
import numpy as np
import pandas as pd
from collections import Counter
from gensim.models.keyedvectors import KeyedVectors

from nltk.corpus import stopwords

class Dictionary( object ):
    """
    from_embedding: Initializes vocab from embedding file. Useful when word embeddings are trained
    on the dataset
    """
    def __init__(self, vocab_size, emb_path, from_embedding=False, stop_words_file=''):
        PAD = 0
        UNK = 1
        self.stop_words_file = stop_words_file
        self._emb_path = emb_path
        self._from_emb = from_embedding
        self._words = []
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': PAD, '<UNK>': UNK}
        self.idx2word = {PAD: '<PAD>', UNK: '<UNK>'}

    def add_word(self, word):
        self._words.append(word)

    def build_vocabulary(self):
        if not self._from_emb:
            cnt = Counter(self._words)
            for idx, (word, _) in enumerate(cnt.most_common(self.vocab_size-2), 2):   #-2 account for PAD and UNK tokens
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        else:
            try:
                sw = codecs.open(self.stop_words_file, 'r', 'utf-8').read().splitlines()
            except:
                print('Stop words file not found! Using default English stop words')
                # sw = stopwords.words("english")
                sw = [u"gonna", "said", "'s'"] # `gonna` is a spacy tok.norm_ artifact

            model = KeyedVectors.load_word2vec_format(self._emb_path)
            for idx, word in enumerate(model.index2word, 2):
                if word not in sw:
                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word[len(self.word2idx)-1] = word

        self.vocab_size = len(self.word2idx)

    def __len__(self):
        return len( self.word2idx )


class Corpus( object ):
    def __init__(self, args):

        self.args = args
        logging.info('Building dictionary...')
        self.dictionary = Dictionary(args.vocab_size, args.emb_path, from_embedding=args.from_emb, stop_words_file=self.args.stop_words)
        if not args.from_emb:
            self.build_dictionary( args.data_dir)
        self.dictionary.build_vocabulary()
        args.vocab_size = len(self.dictionary)

        self.train, self.train_labels, self.train_lens = self.tokenize(args.data_dir, 'train.csv')
        self.test, self.test_labels, self.test_lens = self.tokenize(args.data_dir, 'test.csv')
        self.val, self.val_labels, self.val_lens = self.tokenize(args.data_dir, 'val.csv')

    def build_dictionary( self, data_dir ):
        """Tokenizes a csv file."""

        try:
            sw = codecs.open(self.args.stop_words, 'r', 'utf-8').read().splitlines()
        except:
            print('Stop words file not found! Using default English stop words')
            sw = stopwords.words("english")
            sw += [u"gonna", "said", "'s'"] # `gonna` is a spacy tok.norm_ artifact

        splits = ['train.csv', 'test.csv', 'val.csv']
        for split in splits:
            assert os.path.exists( os.path.join(data_dir, split) )
            # Add words to the dictionary
            df = pd.read_csv(os.path.join(data_dir, split))

            for idx, record in df.iterrows():
                # if idx == 10:   #TODO: remove debug statement
                #     break;
                words = record['text'].split(' ')
                for word in words:
                    if word not in sw:
                        self.dictionary.add_word( word )

    def tokenize(self, data_dir, file):
        # Tokenize file content
        df = pd.read_csv(os.path.join(data_dir, file))

        data = np.zeros((len(df), self.args.max_sent, self.args.max_word), dtype=float)

        labels = np.zeros((len(df)), dtype=float)

        for idx, record in df.iterrows():
            labels[ idx ] = int( record['label'] )

            words = [sent.split(' ') for sent in record['text'].split('<s>')]

            for i, sent in enumerate( words ):
                for j, word in enumerate(sent):
                    if i<self.args.max_sent and j<self.args.max_word:
                        if word not in self.dictionary.word2idx:
                            word = '<UNK>'
                        data[ idx, i, j] = self.dictionary.word2idx[word]

        data = data.reshape((data.shape[0]*data.shape[1], -1))
        zero_idx = np.where(~data.any(axis=1))[0]


        #get lengths
        lengths =  np.count_nonzero(data, axis=1)
        data = data.reshape((len(df), self.args.max_sent, -1))
        lengths = lengths.reshape((len(df), self.args.max_sent, 1))

        return torch.from_numpy(data).long(), torch.from_numpy(labels).long(), torch.from_numpy(lengths).long()
