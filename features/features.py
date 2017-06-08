import os
import sys

sys.path.insert(0, os.path.abspath('..'))
try:
    from features.embeddings import Embeddings
    from features.collection import Collection
    from features.topdocs import Topdocs
except:
    raise


class Features(Embeddings, Collection, Topdocs):
    def __init__(self, parameters):
        Collection.__init__(self, parameters)
        Embeddings.__init__(self)
        Topdocs.__init__(self, parameters)
        self.feature_functions = {}

    def linear_combination(self, term, feature_names, features_weights, feature_parameters):

        self.feature_functions.update({
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

            "td_norm_unigram_count": Topdocs.td_unigram_norm_term_count,
            "td_norm_unigram_document_count": Topdocs.td_unigram_norm_document_count,
            "td_uw_expression_norm_count": Topdocs.td_uw_expression_norm_count,
            "td_od_expression_norm_count": Topdocs.td_od_expression_norm_count,
            "td_uw_expression_norm_document_count": Topdocs.td_uw_expression_norm_document_count,
            "td_od_expression_norm_document_count": Topdocs.td_od_expression_norm_document_count,

        })

        score = 0
        for feature_name in feature_names:
            feature_parameters_ = feature_parameters[feature_name]
            score_ = self.feature_functions[feature_name](self, term, feature_parameters_)
            weight_ = features_weights[feature_name]
            score += weight_ * score_
        return score
