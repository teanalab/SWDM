import sys
from unittest import TestCase
from unittest.mock import MagicMock

from collection.gensim_corpus import GensimCorpus
from parameters.parameters import Parameters


class TestGensimCorpus(TestCase):
    def setUp(self):
        self.parameters = Parameters()
        self.parameters.params["repo_dir"] = '../index/test_files/index'

    def test_get_texts(self):
        GensimCorpus.read_collection = MagicMock(return_value='1312 b c\n a a b')
        corpus = GensimCorpus(self.parameters)
        print(corpus.dictionary, file=sys.stderr)
        print(len(corpus.dictionary), file=sys.stderr)
        self.assertEqual(len(corpus.dictionary), 3)

    def test_read_collection(self):
        self.parameters.params["repo_dir"] = '/scratch/index/indri_5_7/ap8889'
        corpus = GensimCorpus(self.parameters)
        self.assertEqual(len(corpus.dictionary), 185121)
