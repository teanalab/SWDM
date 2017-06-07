import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from embeddings.similarity.neighborhood import Neighborhood
from embeddings.word2vec import Word2vec

matplotlib.rcParams['backend'] = "Qt4Agg"


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

    def run(self, doc_words, file_name, unigrams_color):

        neighborhood = Neighborhood(self.word2vec_model)

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

        colors = sns.color_palette("hls", len(unigrams_color))

        for j, unigram_color in enumerate(unigrams_color):
            for i, txt in enumerate(list(doc_words_model.keys())):
                neighbor_color = neighborhood.find_nearest_neighbor_in_a_list(unigram=unigram_color,
                                                                              other_unigrams=doc_words,
                                                                              min_distance=0.3, neighbor_size=10)
                if txt in neighbor_color:
                    ax.annotate(txt, (z[i], y[i]), color=colors[j])

        fig.set_size_inches(36, 20)
        plt.savefig(file_name, dpi=100)

    # https://github.com/lukas-krecan/ml_examples/blob/master/word2vec_demo.ipynb:
    def run_pca(self, doc_words, file_name, unigrams_color):

        neighborhood = Neighborhood(self.word2vec_model)

        doc_words = [w for w in doc_words if w in self.word2vec_model.vocab]

        embeddings = [self.word2vec_model[w] for w in doc_words]

        print("embeddings =", len(embeddings), file=sys.stderr)

        pca = PCA(n_components=2)
        two_d_embeddings = pca.fit_transform(embeddings)

        colors = sns.color_palette("hls", len(unigrams_color))

        fig, ax = plt.subplots()
        fig.set_size_inches(18, 10)
        for j, unigram_color in enumerate(unigrams_color):
            neighbor_color = neighborhood.find_nearest_neighbor_in_a_list(unigram=unigram_color,
                                                                          other_unigrams=doc_words,
                                                                          min_distance=0.3, neighbor_size=10)
            for label in neighbor_color:
                if label in doc_words:
                    i = doc_words.index(label)
                    x, y = two_d_embeddings[i, :]
                    ax.scatter(x, y, color=colors[j])
                    ax.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom',
                                color=colors[j])
        plt.savefig(file_name, dpi=100)


if __name__ == '__main__':
    tsne_ = Tsne()





    # tsne_.initialize()
    # tsne_.run()
