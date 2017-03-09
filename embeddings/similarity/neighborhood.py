class Neighborhood:
    def __init__(self, word2vec_model):
        self.word2vec_model = word2vec_model

    def find_nearest_neighbor_in_a_list(self, unigram, other_unigrams, min_distance, neighbor_size):
        neighbor = []
        if unigram in self.word2vec_model.wv.vocab:
            for other_unigram in other_unigrams:
                if len(neighbor) > neighbor_size:
                    break
                if other_unigram is not unigram and other_unigram not in neighbor and \
                                other_unigram in self.word2vec_model.wv.vocab:
                    sim = self.word2vec_model.similarity(unigram, other_unigram)
                    if sim > min_distance:
                        neighbor += [other_unigram]
        return neighbor

    def find_significant_neighbors(self, other_unigrams, min_distance, neighbor_size):
        significant_neighbors = []
        for other_unigram in other_unigrams:
            if other_unigram in self.word2vec_model.wv.vocab:
                neighbor = self.find_nearest_neighbor_in_a_list(other_unigram, other_unigrams, min_distance,
                                                                neighbor_size)
                if len(neighbor) == neighbor_size:
                    significant_neighbors += [neighbor]
        significant_neighbors = [list(x) for x in set(tuple(x) for x in significant_neighbors)]
        return significant_neighbors
