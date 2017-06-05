from sdm.weights.bigram_weights import BigramWeights


class OrderedBigramWeights(BigramWeights):
    def __init__(self, parameters):
        BigramWeights.__init__(self, parameters)

        self.feature_names.update({
            "od_expression_norm_count",
            "od_expression_norm_document_count",
            "td_od_expression_norm_count",
            "td_od_expression_norm_document_count",
        })

    def compute_weight(self, term, term_dependent_feature_parameters):

        self.feature_parameters.update(self.parameters.params['feature_parameters']['OrderedBigramWeights'])
        self.features_weights.update(self.parameters.params['features_weights']['OrderedBigramWeights'])

        return BigramWeights.compute_weight(self, term, term_dependent_feature_parameters)
