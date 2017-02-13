from embeddings.word2vec import Word2vec


class EmbeddingSpace:
    def __init__(self, parameters):
        self.parameters = parameters
        self.word2vec = Word2vec()

    def initialize(self):
        self.word2vec.pre_trained_google_news_300_model()
        print("word2vec model obtained.")

    def find_unigrams_in_embedding_space_1(self, word2vec, unigram):
        word2vec_threshold = self.parameters.params["word2vec"]["threshold"]
        word2vec_n_max = self.parameters.params["word2vec"]["n_max"]

        unigrams_in_embedding_space = word2vec.gen_similar_words(unigram=unigram, topn=100)
        unigrams_in_embedding_space_pruned = []
        word2vec_n = 1
        for (unigram, distance) in unigrams_in_embedding_space:
            if word2vec_n > word2vec_n_max:
                break
            if distance > word2vec_threshold and "_" not in unigram and unigram.isalpha():
                unigrams_in_embedding_space_pruned += [(str(unigram), distance)]
                word2vec_n += 1
        return unigrams_in_embedding_space_pruned

    def find_unigrams_in_embedding_space(self, text):
        unigrams_l = text.split(' ')
        unigrams_in_embedding_space = []
        for unigram in unigrams_l:
            new_unigrams = self.find_unigrams_in_embedding_space_1(self.word2vec, unigram)
            unigrams_in_embedding_space += [[(unigram, 1)] + new_unigrams]
        return unigrams_in_embedding_space
