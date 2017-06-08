import sys
from unittest import TestCase

from parameters.parameters import Parameters
from unigrams.original import Original

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 6 / 8 / 17


class TestOriginal(TestCase):
    def setUp(self):
        self.parameters = Parameters()
        self.parameters.params["repo_dir"] = '../index/test_files/index'
        self.original = Original(self.parameters)

    def test_find_unigrams(self):
        res = list(self.original.find_unigrams(
            "Lorem ipsum dolor sit amet, consectetur adipiscing1 elit you i."))
        print(res, file=sys.stderr)
        self.assertEquals(res,
                          [[('Lorem', 1)], [('ipsum', 1)], [('dolor', 1)], [('sit', 1)], [('amet', 1)],
                           [('consectetur', 1)], [('elit', 1)]])
