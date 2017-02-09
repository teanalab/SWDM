from unittest import TestCase

import features.collection


class TestCollection(TestCase):
    def setUp(self):
        self.features = features.collection.Collection('../index/test_files/index')
        self.feature_parameters = {
            "uw_expression_count": {
                "window_size": 17
            },
            "od_expression_count": {
                "window_size": 4
            },
            "uw_expression_document_count": {
                "window_size": 4
            },
            "od_expression_document_count": {
                "window_size": 4
            },
        }

    def test_uw_expression_count(self):
        self.assertEqual(self.features.uw_expression_count("SAMPSON Dog",
                                                           self.feature_parameters["uw_expression_count"]), 2)

    def test_od_document_expression_count(self):
        self.assertEqual(self.features.od_expression_document_count(
            "SAMPSON True", self.feature_parameters["od_expression_document_count"]), 1)

    def test_uw_document_expression_count(self):
        self.assertEqual(self.features.uw_expression_document_count(
            "SAMPSON True", self.feature_parameters["uw_expression_document_count"]), 1)

    def test_term_count(self):
        self.assertEqual(self.features.term_count("dog", {}), 2)

    def test_document_count(self):
        self.assertEqual(self.features.document_count("dog", {}), 1)

    def test_uw_expression_norm_count(self):
        self.assertEqual(self.features.uw_expression_norm_count(
            "SAMPSON Dog", self.feature_parameters["uw_expression_count"]), 0.00273224043715847)

    def test_od_expression_norm_document_count(self):
        self.assertEqual(self.features.od_expression_norm_document_count(
            "SAMPSON True", self.feature_parameters["od_expression_document_count"]), 0.3333333333333333)

    def test_uw_document_expression_norm_count(self):
        self.assertEqual(self.features.uw_expression_norm_document_count(
            "SAMPSON True", self.feature_parameters["uw_expression_document_count"]), 0.3333333333333333)

    def test_norm_term_count(self):
        self.assertEqual(self.features.norm_term_count("dog", {}), 0.00273224043715847)

    def test_norm_document_count(self):
        self.assertEqual(self.features.norm_document_count("dog", {}), 0.3333333333333333)
