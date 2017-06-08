import os
import sys

import nltk
from gensim.corpora import TextCorpus

from index.index import Index

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class GensimCorpus(TextCorpus):
    @staticmethod
    def save_corpus(fname, corpus, id2word=None, metadata=False):
        pass

    def __init__(self, parameters):
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        self.parameters = parameters
        self.index_ = Index(self.parameters)
        self.store_collection_if_not_exists()
        input_ = self.parameters.params["lda"]["file_name"]
        super().__init__(input_)

    def read_collection(self):
        collection_lines = ""
        for i in range(1, self.index_.total_count() + 1):
            doc_text = self.index_.obtain_text_of_a_document(i)
            doc_words = self.tokenizer.tokenize(doc_text)
            doc_words = [w.lower() for w in doc_words if w.isalpha() and len(w) > 2]
            doc_words = [w for w in doc_words if w not in self.stop_words]
            doc_words = [self.index_.index.process_term(w) for w in doc_words]
            collection_lines += ' '.join(doc_words) + '\n'
        return collection_lines

    def store_collection_if_not_exists(self):
        if not os.path.exists(self.parameters.params["lda"]["file_name"]):
            print("reading collection...", file=sys.stderr)
            collection_lines = self.read_collection()
            print("writing collection to:", self.parameters.params["lda"]["file_name"], file=sys.stderr)
            with open(self.parameters.params["lda"]["file_name"], "w") as f:
                f.write(collection_lines)
