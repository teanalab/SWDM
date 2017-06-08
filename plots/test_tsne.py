import json
import os
import re
import sys
from unittest import TestCase

from plots.tsne import Tsne

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class TestTsne(TestCase):
    def setUp(self):
        self.tsne = Tsne()
        sample_json = json.load(open("test_files/wikipedia_page.json"))
        # self.sample_text = sample_json["query"]["pages"]['682482']["extract"]
        with open("../configs/others/human_wiki.txt") as f:
            self.sample_text = f.read()

    def test_run(self):
        self.tsne.initialize_google_news_300()

        doc_words = list(set([i.strip() for i in re.split("; |, |\*|\n| |. ", self.sample_text)]))
        print("doc_words =", len(doc_words), file=sys.stderr)
        file_name = "test_files/tsne.png"

        if os.path.isfile(file_name):
            os.remove(file_name)

        self.tsne.run_pca(doc_words, file_name, ["poach", "wildlife", "preserve"])

        self.assertTrue(os.path.isfile(file_name))
