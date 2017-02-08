import index.index


class Features:
    def __init__(self, repo_dir):
        self.index = index.index.Index(repo_dir)
        self.feature_functions = {
            "uw_expression_count": self.uw_expression_count,
            "od_expression_count": self.od_expression_count
        }

    def uw_expression_count(self, term, feature_parameters):
        return self.index.uw_expression_count(term, feature_parameters["window_size"])

    def od_expression_count(self, term, feature_parameters):
        return self.index.od_expression_count(term, feature_parameters["window_size"])

    def linear_combination(self, term, feature_names, features_weights, feature_parameters):
        score = 0
        for feature_name in feature_names:
            feature_parameters_ = feature_parameters[feature_name]
            score += features_weights[feature_name] * self.feature_functions[feature_name](term, feature_parameters_)
        return score
