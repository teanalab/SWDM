import index.index


class Features:
    def __init__(self, repo_dir):
        self.index = index.index.Index(repo_dir)
        self.feature_functions = {
            "uw_expression_count": self.uw_expression_count,
            "od_expression_count": self.od_expression_count
        }
        self.parameters = {
            "uw_expression_count": {
                "window_size": 17
            },
            "od_expression_count": {
                "window_size": 4
            },
        }

    def uw_expression_count(self, term):
        return self.index.uw_expression_count(term, self.parameters["uw_expression_count"]["window_size"])

    def od_expression_count(self, term):
        return self.index.od_expression_count(term, self.parameters["od_expression_count"]["window_size"])

    def linear_combination(self, term, feature_names, features_weights):
        score = 0
        for feature_name in feature_names:
            score += features_weights[feature_name] * self.feature_functions[feature_name](term)
        return score
