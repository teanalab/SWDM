import features.features
from sdm.weights.weights import Weights


class UnigramWeights(Weights):
    def __init__(self, repo_dir):
        Weights.__init__(self, repo_dir)
        self.features = features.features.Features(repo_dir)
        self.feature_names = {
            "norm_term_count",
            "norm_document_count",
            "unigrams_cosine_similarity_with_orig",
        }
        self.feature_parameters = {
            "norm_term_count": {
            },
            "norm_document_count": {
            },
            "unigrams_cosine_similarity_with_orig": {
            },
        }

        self.features_weights = {
            "norm_term_count": 0.33,
            "norm_document_count": 0.33,
            "unigrams_cosine_similarity_with_orig": 0.33,
        }


    def compute_weight(self, term, term_dependent_feature_parameters):
        self.feature_parameters.update({
            "unigrams_cosine_similarity_with_orig": {
                'unigram_nearest_neighbor': term_dependent_feature_parameters["unigram_nearest_neighbor"]
            }
        })
        return Weights.compute_weight(self, term, term_dependent_feature_parameters)
