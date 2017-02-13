import features.features
from sdm.weights.weights import Weights


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

        terms = term.split(' ')
        cosine_similarity_1 = [item for item in unigram_nearest_neighbor_1 if item[0] == terms[0]][0][1]
        cosine_similarity_2 = [item for item in unigram_nearest_neighbor_2 if item[0] == terms[1]][0][1]

        weight = Weights.compute_weight(self, term, term_dependent_feature_parameters)

        if cosine_similarity_1 == 1 and cosine_similarity_2 == 1:
            return weight * (1 - self.parameters.params["expansion_coefficient"])
        else:
            return weight * self.parameters.params["expansion_coefficient"]
