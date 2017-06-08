import sys
from unittest import TestCase

from lda.train import Train
from parameters.parameters import Parameters

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class TestTrain(TestCase):
    def setUp(self):
        self.parameters = Parameters()
        self.parameters.params["repo_dir"] = '/scratch/index/indri_5_7/ap8889'
        self.parameters.params["lda"] = {"num_topics": 100, "file_name": "../configs/lda/ap8889.txt",
                                         "model": "../configs/lda/ap8889.model"}

    def test_run(self):
        train = Train(self.parameters)
        train.add_corpus()
        train.run()
        print(train.lda.print_topics(1), file=sys.stderr)
        self.assertEqual(train.lda.print_topics(1),
                         [(85,
                           '0.028*"republ" + 0.014*"yugoslavia" + 0.011*"haiti" + 0.011*"ethnic" + 0.011*"govern" '
                           '+ 0.008*"kosovo" + 0.007*"yugoslav" + 0.007*"gregg" + 0.006*"haitian" + 0.006*"provinc"')])

    def test_add_corpus(self):
        train = Train(self.parameters)
        train.add_corpus()
        self.assertEqual(len(train.corpus.dictionary), 184080)
