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
            "uw_expression_document_count": Collection.uw_expression_document_count,
            "od_expression_document_count": Collection.od_expression_document_count,
            "term_count": Collection.term_count,
            "document_count": Collection.document_count,
            "uw_expression_norm_count": Collection.uw_expression_norm_count,
            "od_expression_norm_count": Collection.od_expression_norm_count,
            "uw_expression_norm_document_count": Collection.uw_expression_norm_document_count,
            "od_expression_norm_document_count": Collection.od_expression_norm_document_count,
            "norm_term_count": Collection.norm_term_count,
            "norm_document_count": Collection.norm_document_count,
            "unigrams_cosine_similarity_with_orig": Embeddings.unigrams_cosine_similarity_with_orig,
            "bigrams_cosine_similarity_with_orig": Embeddings.bigrams_cosine_similarity_with_orig,
        }

    def linear_combination(self, term, feature_names, features_weights, feature_parameters):
        score = 0
        for feature_name in feature_names:
            feature_parameters_ = feature_parameters[feature_name]
            score_ = self.feature_functions[feature_name](self, term, feature_parameters_)
            weight_ = features_weights[feature_name]
            score += weight_ * score_
        return score
