from embeddings.word2vec import Word2vec


class Neighborhood(Word2vec):

    def __init__(self):
        super().__init__()

    def initialize(self):
        self.pre_trained_google_news_300_model()
        print("word2vec model obtained.")

    def find_nearest_neighbor_in_a_list(self, unigram, other_unigrams, min_distance, neighbor_size):
        neighbor = []
        if unigram in self.model.vocab:
            for other_unigram in other_unigrams:
                if len(neighbor) > neighbor_size:
                    break
                if other_unigram is not unigram and other_unigram not in neighbor and other_unigram in self.model.vocab:
                    sim = self.model.similarity(unigram, other_unigram)
                    if sim > min_distance:
                        neighbor += [other_unigram]
        return neighbor
