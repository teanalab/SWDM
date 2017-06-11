from unittest import TestCase

import features.topdocs
from parameters.parameters import Parameters

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class TestTopdocs(TestCase):
    def setUp(self):
        self.parameters = Parameters()
        self.parameters.params["repo_dir"] = '../index/test_files/index'
        self.topdocs_ = features.topdocs.Topdocs(self.parameters)
        self.feature_parameters = {
            "OrderedBigramWeights": {
                "td_od_expression_norm_count": {
                    "window_size": 1,
                    "n_top_docs": 15
                },
                "td_od_expression_norm_document_count": {
                    "window_size": 1,
                    "n_top_docs": 15
                }
            },
            "UnigramWeights": {
                "td_norm_unigram_count": {
                    "n_top_docs": 15
                },
                "td_norm_unigram_document_count": {
                    "n_top_docs": 15
                }
            },
            "UnorderedBigramWeights": {
                "td_uw_expression_norm_count": {
                    "window_size": 2,
                    "n_top_docs": 15
                },
                "td_uw_expression_norm_document_count": {
                    "window_size": 2,
                    "n_top_docs": 15
                }
            }
        }
        self.topdocs_.init_top_docs_run_query("a")

    def test_td_uw_expression_count(self):
        self.assertEqual(self.topdocs_.td_uw_expression_count("I Will",
                                                              self.feature_parameters["UnorderedBigramWeights"][
                                                                  "td_uw_expression_norm_count"]), 7)

    def test_td_od_expression_count(self):
        self.assertEqual(self.topdocs_.td_od_expression_count("I Will",
                                                              self.feature_parameters["UnorderedBigramWeights"][
                                                                  "td_uw_expression_norm_count"]), 7)

    def test_td_uw_expression_document_count(self):
        self.assertEqual(self.topdocs_.td_uw_expression_document_count("you",
                                                                       self.feature_parameters[
                                                                           "UnorderedBigramWeights"][
                                                                           "td_uw_expression_norm_count"]), 2)
        self.assertEqual(self.topdocs_.td_uw_expression_document_count("I will",
                                                                       self.feature_parameters[
                                                                           "UnorderedBigramWeights"][
                                                                           "td_uw_expression_norm_count"]), 1)

    def test_td_od_expression_document_count(self):
        self.assertEqual(self.topdocs_.td_od_expression_document_count("you",
                                                                       self.feature_parameters[
                                                                           "UnorderedBigramWeights"][
                                                                           "td_uw_expression_norm_count"]), 2)
        self.assertEqual(self.topdocs_.td_od_expression_document_count("I will",
                                                                       self.feature_parameters[
                                                                           "UnorderedBigramWeights"][
                                                                           "td_uw_expression_norm_count"]), 1)

    def test_td_unigram_term_count(self):
        self.assertEqual(self.topdocs_.td_unigram_term_count("you",
                                                             self.feature_parameters[
                                                                 "UnorderedBigramWeights"][
                                                                 "td_uw_expression_norm_count"]), 2)

    def test_td_unigram_document_count(self):
        self.assertEqual(self.topdocs_.td_unigram_document_count("you",
                                                                 self.feature_parameters[
                                                                     "UnorderedBigramWeights"][
                                                                     "td_uw_expression_norm_count"]), 10)

    def test_td_uw_expression_norm_count(self):
        gram_pair = (('two', 'stand'), (('pair', 0.5850129127502441), ('sit', 0.5178298950195312)))
        self.assertEqual(self.topdocs_.td_uw_expression_norm_count(gram_pair,
                                                                   self.feature_parameters[
                                                                       "UnorderedBigramWeights"][
                                                                       "td_uw_expression_norm_count"]),
                         6.595780513961311)

    def test_td_od_expression_norm_count(self):
        gram_pair = (('two', 'stand'), (('pair', 0.5850129127502441), ('sit', 0.5178298950195312)))
        self.assertEqual(self.topdocs_.td_uw_expression_norm_count(gram_pair,
                                                                   self.feature_parameters[
                                                                       "UnorderedBigramWeights"][
                                                                       "td_uw_expression_norm_count"]),
                         6.595780513961311)

    def test_td_uw_expression_norm_document_count(self):
        gram_pair = (('two', 'stand'), (('pair', 0.5850129127502441), ('sit', 0.5178298950195312)))
        self.assertEqual(self.topdocs_.td_uw_expression_norm_document_count(gram_pair,
                                                                            self.feature_parameters[
                                                                                "UnorderedBigramWeights"][
                                                                                "td_uw_expression_norm_count"]),
                         2.70805020110221)

    def test_td_od_expression_norm_document_count(self):
        gram_pair = (('two', 'stand'), (('pair', 0.5850129127502441), ('sit', 0.5178298950195312)))
        self.assertEqual(self.topdocs_.td_od_expression_norm_document_count(gram_pair,
                                                                            self.feature_parameters[
                                                                                "UnorderedBigramWeights"][
                                                                                "td_uw_expression_norm_count"]),
                         2.70805020110221)

    def test_td_unigram_norm_term_count(self):
        gram_pair = ('two', ('pair', 0.5850129127502441))
        self.assertEqual(self.topdocs_.td_unigram_norm_term_count(gram_pair,
                                                                  self.feature_parameters[
                                                                      "UnorderedBigramWeights"][
                                                                      "td_uw_expression_norm_count"]),
                         5.902633333401366)

    def test_td_unigram_norm_document_count(self):
        gram_pair = ('two', ('pair', 0.5850129127502441))
        self.assertEqual(self.topdocs_.td_unigram_norm_document_count(gram_pair,
                                                                      self.feature_parameters[
                                                                          "UnorderedBigramWeights"][
                                                                          "td_uw_expression_norm_count"]),
                         2.0149030205422647)
