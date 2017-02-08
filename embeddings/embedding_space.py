from embeddings.word2vec import Word2vec


class EmbeddingSpace:
    def __init__(self):
        pass

    @staticmethod
    def find_unigrams_in_embedding_space_1(word2vec, unigram, word2vec_threshold):
        unigrams_in_embedding_space = word2vec.gen_similar_words(unigram=unigram, topn=100)
        unigrams_in_embedding_space = [(i.encode('ascii', 'ignore'), j) for (i, j) in unigrams_in_embedding_space if
                                       j > word2vec_threshold]
        return unigrams_in_embedding_space

    def find_unigrams_in_embedding_space(self, text, word2vec_threshold):
        unigrams_l = text.split(' ')
        word2vec = Word2vec()
        word2vec.pre_trained_google_news_300_model()
        unigrams_in_embedding_space = []
        for unigram in unigrams_l:
            new_unigrams = self.find_unigrams_in_embedding_space_1(word2vec, unigram, word2vec_threshold)
            unigrams_in_embedding_space += \
                [[(unigram, 1)] + new_unigrams]
        return unigrams_in_embedding_space
