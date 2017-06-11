from math import log

import index.index

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class Collection:
    def __init__(self, parameters):
        self.index = index.index.Index(parameters)

    @staticmethod
    def gram_pair_2_bigarm(gram_pair):
        return ' '.join([gram_pair[1][0][0], gram_pair[1][1][0]])

    @staticmethod
    def gram_pair_2_unigarm(gram_pair):
        return gram_pair[1][0]

    def uw_expression_count(self, term, feature_parameters):
        return self.index.uw_expression_count(term, feature_parameters["window_size"])

    def od_expression_count(self, term, feature_parameters):
        return self.index.od_expression_count(term, feature_parameters["window_size"])

    def uw_expression_document_count(self, term, feature_parameters):
        return self.index.uw_expression_document_count(term, feature_parameters["window_size"])

    def od_expression_document_count(self, term, feature_parameters):
        return self.index.od_expression_document_count(term, feature_parameters["window_size"])

    def term_count(self, term, feature_parameters):
        del feature_parameters
        return self.index.term_count(term)

    def document_count(self, term, feature_parameters):
        del feature_parameters
        return self.index.document_count(term)

    def uw_expression_norm_count(self, gram_pair, feature_parameters):
        bigram = self.gram_pair_2_bigarm(gram_pair)
        return -log((self.uw_expression_count(bigram, feature_parameters) + 1) / self.index.total_terms())

    def od_expression_norm_count(self, gram_pair, feature_parameters):
        bigram = self.gram_pair_2_bigarm(gram_pair)
        return -log((self.od_expression_count(bigram, feature_parameters) + 1) / self.index.total_terms())

    def uw_expression_norm_document_count(self, gram_pair, feature_parameters):
        bigram = self.gram_pair_2_bigarm(gram_pair)
        return -log((self.uw_expression_document_count(bigram, feature_parameters) + 1) / self.index.total_count())

    def od_expression_norm_document_count(self, gram_pair, feature_parameters):
        bigram = self.gram_pair_2_bigarm(gram_pair)
        return -log((self.od_expression_document_count(bigram, feature_parameters) + 1) / self.index.total_count())

    def norm_term_count(self, gram_pair, feature_parameters):
        unigram = self.gram_pair_2_unigarm(gram_pair)
        return -log((self.term_count(unigram, feature_parameters) + 1) / self.index.total_terms())

    def norm_document_count(self, gram_pair, feature_parameters):
        unigram = self.gram_pair_2_unigarm(gram_pair)
        return -log((self.document_count(unigram, feature_parameters) + 1) / self.index.total_count())
