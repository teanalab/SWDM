__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class Embeddings:
    def __init__(self):
        pass

    @staticmethod
    def is_not_used():
        pass

    def unigrams_cosine_similarity_with_orig(self, unigram_pair, feature_parameters):
        self.is_not_used()
        del feature_parameters
        return unigram_pair[1][1]

    def bigrams_cosine_similarity_with_orig(self, bigram_pair, feature_parameters):
        self.is_not_used()
        del feature_parameters
        cosine_similarity_1 = bigram_pair[1][0][1]
        cosine_similarity_2 = bigram_pair[1][1][1]
        return (cosine_similarity_1 + cosine_similarity_2) / 2
