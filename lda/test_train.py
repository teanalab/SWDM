import sys
from unittest import TestCase

from lda.train import Train
from parameters.parameters import Parameters


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
                         [(29,
                           '0.036*"bank" + 0.024*"loan" + 0.020*"savings" + 0.018*"said" + 0.018*"banks" + '
                           '0.017*"federal" + 0.015*"the" + 0.013*"loans" + 0.012*"wright" + 0.011*"institutions"')])

    def test_add_corpus(self):
        train = Train(self.parameters)
        train.add_corpus()
        self.assertEqual(len(train.corpus.dictionary), 184080)
