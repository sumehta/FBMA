import random
import logging
import torch
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def get_embeddings(emb_path, dictionary, dim=100):
    '''
    Get embeddings from a pre-trained embeddings file
    '''
    emb = np.zeros((len(dictionary), dim), dtype=float)

    if emb_path.split('/')[-1].split('.')[-1] != 'txt':  #gensim case (ends with .w2v)
        model =  KeyedVectors.load_word2vec_format(emb_path)
        for word, idx in dictionary.word2idx.items():
            if idx not in [0, 1]: #skip '<PAD>' and '<UNK>'
                emb[idx, :] = model.get_vector(word)
    else:
        with open(emb_path, 'r') as f:
            embeddings = f.readlines()

        embeddings = map(lambda x: x.strip('\n').strip(' ').split(' '), embeddings)
        for line in embeddings:
            if len(line)==2:  # wiki embeddings first line
                continue
            if line[0] in dictionary.word2idx:
                emb[dictionary.word2idx[line[0]], :] = np.asarray(map(lambda x: float(x), line[1:]))

    return torch.from_numpy(emb)

class CrossValidationSplitsTitle(object):
    def __init__(self, data, labels, lens, titles, split=0.20):
        self.data = data
        self.labels = labels
        self.lens = lens
        self.titles = titles
        self.split = split
        self.train, self.train_labels, self.train_lens, self.train_titles, self.val, self.val_labels, self.val_lens, self.val_titles = self._generate_splits()

    def _generate_splits(self):
        shuffle = torch.randperm(len(self.data)) # bsz
        data = self.data[shuffle] # shuffled data
        labels = self.labels[shuffle] # shuffled labels
        lens = self.lens[shuffle] # shuffled labels
        titles = self.titles[shuffle] # shuffled titles

        num_train = int(len(data) - len(data)*self.split)

        train = data[: num_train]
        train_labels = labels[: num_train]
        train_lens = lens[: num_train]
        train_titles = titles[: num_train]

        val = data[num_train :]
        val_labels = labels[num_train :]
        val_lens = lens[num_train :]
        val_titles = titles[num_train :]


        return train, train_labels, train_lens, train_titles, val, val_labels, val_lens, val_titles


class Batchify(object):
    def __init__(self, data, labels, lens, bsz=32, cuda=False):
        indices = np.random.permutation(len(data))
        self.data = data[indices]
        self.labels = labels[indices]
        self.lens = lens[indices]
        self.num_batches = data.size(0) // bsz
        self.batch_size = bsz

        if cuda:
            self.data = self.data.cuda()
            self.labels = self.labels.cuda()
            self.lens = self.lens.cuda()

    def next(self):
        i = 0
        assert self.data.size(0) == self.labels.size(0)
        batches_left = self.num_batches
        while batches_left > 0:
            yield self.data[i:i+self.batch_size], self.labels[i:i+self.batch_size], self.lens[i:i+self.batch_size]
            i += self.batch_size
            batches_left -= 1


class CrossValidationSplits(object):
    def __init__(self, data, labels, lens, split=0.20):
        self.data = data
        self.labels = labels
        self.lens = lens
        self.split = split
        self.train, self.train_labels, self.train_lens, self.val, self.val_labels, self.val_lens = self._generate_splits()

    def _generate_splits(self):
        shuffle = torch.randperm(len(self.data)) # bsz
        data = self.data[shuffle] # shuffled data
        labels = self.labels[shuffle] # shuffled labels
        lens = self.lens[shuffle] # shuffled labels

        num_train = int(len(data) - len(data)*self.split)
        train = data[: num_train]
        train_labels = labels[: num_train]
        train_lens = lens[: num_train]
        val = data[num_train :]
        val_labels = labels[num_train :]
        val_lens = lens[num_train :]


        return train, train_labels, train_lens, val, val_labels, val_lens

class Metrics(object):
    def __init__(self, nclass, labels, pos_label=1):
        self._nclass = nclass
        self._labels = labels
        self._pos_label = pos_label

    def accuracy(self, target, pred, average='binary'):
        return accuracy_score(target, pred)

    def precision(self, target, pred, average='binary'):
        return precision_score(target, pred, average=average, labels = self._labels)

    def recall(self, target, pred, average='binary'):
        return recall_score(target, pred, average=average, labels = self._labels)

    def f1(self, target, pred, average='binary'):
        return f1_score(target, pred, average=average, labels = self._labels)
