from unittest import TestCase

import index.index
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

    def test_document_count(self):
        self.assertEqual(self.index_.document_count("dog"), 1)

    def test_check_if_have_same_stem(self):
        self.assertEqual(self.index_.check_if_have_same_stem("goes", "goe"), True)
        self.assertEqual(self.index_.check_if_have_same_stem("goes", "g"), False)
