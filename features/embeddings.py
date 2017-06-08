__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class Embeddings:
    def __init__(self):
        pass

    @staticmethod
    def is_not_used():
        pass

    def unigrams_cosine_similarity_with_orig(self, term, feature_parameters):
        self.is_not_used()
        unigram_nearest_neighbor = feature_parameters['unigram_nearest_neighbor']
        return [item for item in unigram_nearest_neighbor if item[0] == term][0][1]

    def bigrams_cosine_similarity_with_orig(self, term, feature_parameters):
        self.is_not_used()
        terms = term.split(' ')
        unigram_nearest_neighbor_1 = feature_parameters['unigram_nearest_neighbor_1']
        unigram_nearest_neighbor_2 = feature_parameters['unigram_nearest_neighbor_2']
        cosine_similarity_1 = [item for item in unigram_nearest_neighbor_1 if item[0] == terms[0]][0][1]
        cosine_similarity_2 = [item for item in unigram_nearest_neighbor_2 if item[0] == terms[1]][0][1]
        return (cosine_similarity_1 + cosine_similarity_2) / 2
