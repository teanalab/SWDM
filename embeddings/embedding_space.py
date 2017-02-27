import sys
from collections import defaultdict

from embeddings.word2vec import Word2vec
from index.index import Index


class EmbeddingSpace:
    def __init__(self, parameters):
        self.parameters = parameters
        self.word2vec = Word2vec()
        self.index_ = Index(self.parameters)
        self.unigrams_in_embedding_space_history = defaultdict()

    def initialize(self):
        self.word2vec.pre_trained_google_news_300_model()
        print("word2vec model obtained.")

    def check_if_already_stem_exists(self, unigrams_in_embedding_space_pruned, unigram, orig_unigram):
        if self.index_.check_if_have_same_stem(unigram, orig_unigram):
            return True
        for (other_unigram, distance) in unigrams_in_embedding_space_pruned:
            if self.index_.check_if_have_same_stem(other_unigram, unigram):
                return True
        return False

    def gen_similar_words(self, orig_unigram, word2vec):
        if orig_unigram not in self.unigrams_in_embedding_space_history:
            unigrams_in_embedding_space = word2vec.gen_similar_words(unigram=orig_unigram, topn=100)
            self.unigrams_in_embedding_space_history[orig_unigram] = unigrams_in_embedding_space
            return unigrams_in_embedding_space
        else:
            return self.unigrams_in_embedding_space_history[orig_unigram]

    def find_unigrams_in_embedding_space_1(self, word2vec, orig_unigram):
        word2vec_threshold = self.parameters.params["word2vec"]["threshold"]
        word2vec_n_max = self.parameters.params["word2vec"]["n_max"]

        unigrams_in_embedding_space = self.gen_similar_words(orig_unigram, word2vec)

        unigrams_in_embedding_space_pruned = []
        word2vec_n = 1
        for (unigram, distance) in unigrams_in_embedding_space:
            unigram = str(unigram)
            if word2vec_n > word2vec_n_max:
                break
            if distance > word2vec_threshold and "_" not in unigram and unigram.isalpha() and not \
                    self.index_.check_if_have_same_stem(orig_unigram, unigram):
                # not self.check_if_already_stem_exists(unigrams_in_embedding_space_pruned, unigram, orig_unigram):
                unigrams_in_embedding_space_pruned += [(unigram, distance)]
                word2vec_n += 1
        return unigrams_in_embedding_space_pruned

    def find_unigrams_in_embedding_space(self, text):
        unigrams_l = text.split(' ')
        unigrams_in_embedding_space = []
        for unigram in unigrams_l:
            new_unigrams = self.find_unigrams_in_embedding_space_1(self.word2vec, unigram)
            unigrams_in_embedding_space += [[(unigram, 1)] + new_unigrams]
        return unigrams_in_embedding_space
