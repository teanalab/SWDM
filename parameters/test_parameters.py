from unittest import TestCase

from parameters.parameters import Parameters


class TestParameters(TestCase):

    def setUp(self):
        self.parameters = Parameters()
        self.parameters.params_file = "test_files/parameters.json"

    def test_write_to_parameters_file(self):
        params_ = {
            "norm_term_count": {
            },
            "norm_document_count": {
            },
            "unigrams_cosine_similarity_with_orig": {
            },
        }
        self.parameters.params = params_
        self.parameters.write_to_parameters_file(self.parameters.params_file)
        self.parameters.read_from_params_file()
        self.assertEqual(params_, self.parameters.params)
