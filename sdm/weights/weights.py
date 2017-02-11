import sys

import features.features


class Weights:
    def __init__(self, repo_dir):
        self.features = features.features.Features(repo_dir)

        self.feature_names = set()

        self.features_weights = dict()

        self.feature_parameters = dict()

    def compute_weight(self, term, term_dependent_feature_parameters):
        del term_dependent_feature_parameters
        print(len((term, self.feature_names, self.features_weights,
                                                self.feature_parameters)), file=sys.stderr)
        return self.features.linear_combination(term, self.feature_names, self.features_weights,
                                                self.feature_parameters)
