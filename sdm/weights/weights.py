import features.features


class Weights:
    def __init__(self, parameters):
        self.features = features.features.Features(parameters)

        self.feature_names = set()

        self.features_weights = dict()

        self.feature_parameters = dict()

    def init_top_docs_run_query(self, query):
        self.features.init_top_docs_run_query(query)

    def compute_weight(self, term, term_dependent_feature_parameters):
        del term_dependent_feature_parameters
        return self.features.linear_combination(term, self.feature_names, self.features_weights,
                                                self.feature_parameters)
