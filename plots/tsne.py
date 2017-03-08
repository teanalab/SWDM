import re

import sys
from collections import defaultdict

from gensim.models import Word2Vec
from sklearn.datasets import fetch_20newsgroups

from embeddings.word2vec import Word2vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class Tsne():
    def __init__(self):
        self.word2vec = Word2vec()
        self.word2vec_model = None

    def initialize_google_news_300(self):
        self.word2vec.pre_trained_google_news_300_model()
        print("word2vec model obtained.")
        self.word2vec_model = self.word2vec.model

    # from http://stackoverflow.com/a/42261870/4642669
    @staticmethod
    def clean_text(text):
        """Remove posting header, split by sentences and words, keep only letters"""
        lines = re.split('[?!.:]\s', re.sub('^.*Lines: \d+', '', re.sub('\n', ' ', text)))
        return [re.sub('[^a-zA-Z]', ' ', line).lower().split() for line in lines]

    def initialize_google_20newsgroups(self):
        train = fetch_20newsgroups()
        sentences = [line for text in train.data for line in self.clean_text(text)]
        self.word2vec_model = Word2Vec(sentences, workers=4, size=100, min_count=50, window=10, sample=1e-3)
        print("word2vec model obtained.")

    def run(self, doc_words, file_name):

        doc_words_model = {k: self.word2vec_model.wv.vocab[k] for k in list(self.word2vec_model.wv.vocab.keys())
                           if k in doc_words}

        print("doc_words_model =", len(doc_words_model), file=sys.stderr)

        X = []
        for k, v in doc_words_model.items():
            X += [self.word2vec_model[{k: v}][0]]

        print("X =", X, file=sys.stderr)

        tsne = TSNE(n_components=2, random_state=0)
        X_tsne = tsne.fit_transform(X)

        print("X_tsne =", len(X_tsne), file=sys.stderr)

        z = X_tsne[:, 0]
        y = X_tsne[:, 1]
        fig, ax = plt.subplots()
        ax.scatter(z, y)

        for i, txt in enumerate(list(doc_words_model.keys())):
            ax.annotate(txt, (z[i], y[i]))

        fig.set_size_inches(36, 20)
        plt.savefig(file_name, dpi=100)


if __name__ == '__main__':
    tsne_ = Tsne()





    # tsne_.initialize()
    # tsne_.run()
