import features.features


class UnigramWeights:
    def __init__(self):
        self.features = features.features.Features
        self.feature_names = {
            "norm_term_count",
            "norm_document_count",
            "unigrams_cosine_similarity_with_orig",
        }
        self.feature_parameters = {
            "uw_expression_count": {
                "window_size": 17
            },
            "od_expression_count": {
                "window_size": 4
            },
            "uw_expression_document_count": {
                "window_size": 4
            },
            "od_expression_document_count": {
                "window_size": 4
            },
        }

    def compute_weight(self):
        self.features.linear_combination(feature_names, features_weights, feature_parameters)
