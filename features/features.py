import os
import sys

sys.path.insert(0, os.path.abspath('..'))
try:
    from features.embeddings import Embeddings
    from features.collection import Collection
except:
    raise


class Features(Embeddings, Collection):
    def __init__(self, repo_dir):
        Collection.__init__(self, repo_dir)
        Embeddings.__init__(self)
        self.feature_functions = {
            "uw_expression_count": Collection.uw_expression_count,
            "od_expression_count": Collection.od_expression_count,
            "uw_document_expression_count": Collection.uw_document_expression_count,
            "od_document_expression_count": Collection.od_document_expression_count,
            "term_count": Collection.term_count,
            "document_count": Collection.document_count,
            "euclidean_distance_from_orig": Embeddings.cosine_similarity_with_orig,
        }

    def linear_combination(self, term, feature_names, features_weights, feature_parameters):
        score = 0
        for feature_name in feature_names:
            feature_parameters_ = feature_parameters[feature_name]
            score += features_weights[feature_name] * self.feature_functions[feature_name](self, term,
                                                                                           feature_parameters_)
        return score