from unittest import TestCase

from parameters.parameters import Parameters
from queries.queriesEvaluator import QueriesEvaluator

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class TestQueriesEvaluator(TestCase):
    def setUp(self):
        self.parameters = Parameters()
        pass

    def test_check_indri_run_query_exception(self):
        with self.assertRaises(Exception) as context:
            QueriesEvaluator(self.parameters).check_indri_run_query_exception("test_files/indri.runs")

        self.assertTrue('IndriRunQuery returned an EXCEPTION' in str(context.exception))

        self.assertEqual(True,
                         QueriesEvaluator(self.parameters).check_indri_run_query_exception("test_files/indri_.runs"))
