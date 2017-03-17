import sys

from gensim.corpora import TextCorpus

from index.index import Index

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from gensim import utils


class GensimCorpus(TextCorpus):
    def __init__(self, parameters):
        self.index_ = Index(parameters)
        self.collection_lines = self.read_collection()
        input = StringIO(self.collection_lines)
        super().__init__(input)

    def read_collection(self):
        collection_lines = ""
        for i in range(1, self.index_.total_count() + 1):
            collection_lines += ' '.join(self.index_.obtain_terms_of_a_document(i)[1]) + '\n'

        return collection_lines

    def getstream(self):
        return self.input

    def get_texts(self):
        with self.getstream() as lines:
            for lineno, line in enumerate(lines):
                if self.metadata:
                    yield utils.tokenize(line, lowercase=True), (lineno,)
                else:
                    yield utils.tokenize(line, lowercase=True)
