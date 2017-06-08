import sys

import gensim

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class Word2vec:
    def __init__(self):
        self.model = None
        pass

    def pre_trained_google_news_300_model(self):
        self.model = gensim.models.KeyedVectors.load_word2vec_format('/scratch/data/GoogleNews-vectors-negative300.bin',
                                                                     binary=True)

    @staticmethod
    def sentences(fname):
        sentences = []
        for line in open(fname):
            print(line, file=sys.stderr)
            sentences += [line.strip().split()]
        return sentences

    def gen_similar_words(self, unigram, topn):
        try:
            res = self.model.most_similar([unigram], [], topn)
        except KeyError:
            res = []
        return res


if __name__ == '__main__':
    pass
