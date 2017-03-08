import json
from unittest import TestCase

import sys

from plots.tsne import Tsne


class TestTsne(TestCase):

    def setUp(self):
        self.tsne = Tsne()
        sample_json = json.load(open("test_files/wikipedia_page.json"))
        self.sample_text = sample_json["query"]["pages"]['682482']["extract"]

    def test_run(self):
        self.tsne.initialize_google_news_300()

        doc_words = [i.strip() for i in self.sample_text.split(' ')]
        print(doc_words, file=sys.stderr)
        print("doc_words =", len(doc_words), file=sys.stderr)
        file_name = "test_files/tsne.png"
        self.tsne.run(doc_words, file_name)
