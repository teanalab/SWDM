from sdm.weights.bigram_weights import BigramWeights


class UnorderedBigramWeights(BigramWeights):
    def __init__(self, repo_dir):
        BigramWeights.__init__(self, repo_dir)

        self.feature_names.update({
            "uw_expression_norm_count",
            "uw_expression_norm_document_count",
        })

        self.feature_parameters.update({
            "uw_expression_norm_count": {
                "window_size": 17
            },
            "uw_expression_norm_document_count": {
                "window_size": 17
            }
        })

        self.features_weights = {
            "uw_expression_norm_count": 0.33,
            "uw_expression_norm_document_count": 0.33,
            "bigrams_cosine_similarity_with_orig": 0.33,
        }
