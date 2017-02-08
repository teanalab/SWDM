from unittest import TestCase

import index.index


class TestIndex(TestCase):
    def setUp(self):
        self.index_ = index.index.Index('test_files/index')

    def test_uw_expression_count(self):
        self.assertEqual(self.index_.uw_expression_count("SAMPSON Dog", 12), 2)

    def test_od_expression_count(self):
        self.assertEqual(self.index_.od_expression_count("SAMPSON True", 12), 1)
