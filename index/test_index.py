from unittest import TestCase

import sys

import index.index
from embeddings.similarity.neighborhood import Neighborhood
from parameters.parameters import Parameters


class TestIndex(TestCase):
    def setUp(self):
        self.parameters = Parameters()
        self.parameters.params["repo_dir"] = '../index/test_files/index'

        self.index_ = index.index.Index(self.parameters)

    def test_uw_expression_count(self):
        self.assertEqual(self.index_.uw_expression_count("SAMPSON Dog", 12), 2)

    def test_od_expression_count(self):
        self.assertEqual(self.index_.od_expression_count("SAMPSON True", 12), 1)

    def test_uw_document_expression_count(self):
        self.assertEqual(self.index_.uw_expression_document_count("SAMPSON True", 12), 1)

    def test_od_document_expression_count(self):
        self.assertEqual(self.index_.od_expression_document_count("SAMPSON True", 12), 1)

    def test_term_count(self):
        self.assertEqual(self.index_.term_count("dog"), 2)

        self.parameters.params["repo_dir"] = '/scratch/index/indri_5_7/ap8889'
        self.index_ = index.index.Index(self.parameters)

        self.assertEqual(self.index_.term_count("emotional"), 3515)

    def test_document_count(self):
        self.assertEqual(self.index_.document_count("dog"), 1)

        self.parameters.params["repo_dir"] = '/scratch/index/indri_5_7/ap8889'
        self.index_ = index.index.Index(self.parameters)

        self.assertEqual(self.index_.document_count("emotional"), 2973)

    def test_check_if_have_same_stem(self):
        self.assertEqual(self.index_.check_if_have_same_stem("goes", "goe"), True)
        self.assertEqual(self.index_.check_if_have_same_stem("goes", "g"), False)
        self.assertEqual(self.index_.check_if_have_same_stem("first", "mr"), False)

    def test_idf(self):
        self.assertEqual(self.index_.idf("dog"), 0.4054651081081644)

    def test_tfidf(self):
        self.parameters.params["repo_dir"] = '/scratch/index/indri_5_7/ap8889'

        self.index_ = index.index.Index(self.parameters)
        doc_words = Neighborhood(None, self.parameters).get_words("../configs/others/pride_and_prejudice_wiki.txt")
        tfidf_1 = self.index_.tfidf('emotional', doc_words)
        print(tfidf_1, file=sys.stderr)
        tfidf_2 = self.index_.tfidf('is', doc_words)
        print(tfidf_2, file=sys.stderr)

        self.assertEqual(tfidf_1, 0)
        self.assertEqual(tfidf_2, 0)

    def test_tf(self):
        doc_words = Neighborhood(None, self.parameters).get_words("../configs/others/pride_and_prejudice_wiki.txt")

        self.assertEqual(self.index_.tf("dog", doc_words), 0.5)

        self.parameters.params["repo_dir"] = '/scratch/index/indri_5_7/ap8889'
        self.index_ = index.index.Index(self.parameters)

        self.assertEqual(self.index_.tf("emotional", doc_words), 0.5096153846153846)

    def test_check_if_exists_in_index(self):
        self.parameters.params["repo_dir"] = '/scratch/index/indri_5_7/ap8889'
        self.index_ = index.index.Index(self.parameters)

        self.assertTrue(self.index_.check_if_exists_in_index("emotional"))
        self.assertFalse(self.index_.check_if_exists_in_index("first"))
