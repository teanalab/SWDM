import os

import gensim


class Word2vec:
    def __init__(self):
        self.model = None
        pass

    def setup_google_news_300_model(self):
        self.model = gensim.models.Word2Vec.load_word2vec_format('/scratch/data/GoogleNews-vectors-negative300.bin',
                                                                 binary=True)

    def train_model_from_sentences(self, sentences):
        # train word2vec on the two sentences
        self.model = gensim.models.Word2Vec(sentences, min_count=1)

    # used https://rare-technologies.com/word2vec-tutorial/ :
    class Sentences(object):
        def __init__(self, fname):
            self.fname = fname

        def __iter__(self):
            for line in open(self.fname):
                yield line.split()

    def gen_similar_words(self, unigram, topn):
        res = self.model.most_similar([unigram], [], topn)
        return res


if __name__ == '__main__':
    pass
