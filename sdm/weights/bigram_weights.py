import features.features
from sdm.weights.weights import Weights

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class BigramWeights(Weights):
    def __init__(self, parameters):
        self.parameters = parameters
        Weights.__init__(self, self.parameters)
        self.features = features.features.Features(self.parameters)

        self.feature_names.update({
            "bigrams_cosine_similarity_with_orig"
        })

        self.features_weights.update({})

        self.feature_parameters.update({})

    def compute_weight(self, term, term_dependent_feature_parameters):
        unigram_nearest_neighbor_1 = term_dependent_feature_parameters["unigram_nearest_neighbor_1"]
        unigram_nearest_neighbor_2 = term_dependent_feature_parameters["unigram_nearest_neighbor_2"]
        self.feature_parameters.update({
            "bigrams_cosine_similarity_with_orig": {
                'unigram_nearest_neighbor_1': unigram_nearest_neighbor_1,
                'unigram_nearest_neighbor_2': unigram_nearest_neighbor_2
            }
        })

        return Weights.compute_weight(self, term, term_dependent_feature_parameters)
