from sdm.weights.bigram_weights import BigramWeights


class OrderedBigramWeights(BigramWeights):
    def __init__(self, repo_dir):
        BigramWeights.__init__(self, repo_dir)

        self.feature_names.update({
            "od_expression_norm_count",
            "od_expression_norm_document_count",
        })

        self.feature_parameters.update({
            "od_expression_norm_count": {
                "window_size": 17
            },
            "od_expression_norm_document_count": {
                "window_size": 17
            }
        })

        self.features_weights = {
            "od_expression_norm_count": 0.33,
            "od_expression_norm_document_count": 0.33,
            "bigrams_cosine_similarity_with_orig": 0.33,
        }

    def test(self):
        pass