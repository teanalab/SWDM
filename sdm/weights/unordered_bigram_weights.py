from sdm.weights.bigram_weights import BigramWeights


class UnorderedBigramWeights(BigramWeights):
    def __init__(self, parameters):
        self.parameters = parameters
        BigramWeights.__init__(self, self.parameters)

        self.feature_names.update({
            "uw_expression_norm_count",
            "uw_expression_norm_document_count",
        })

        self.feature_parameters.update(self.parameters.params['feature_parameters']['UnorderedBigramWeights'])

        self.features_weights.update(self.parameters.params['features_weights']['UnorderedBigramWeights'])
