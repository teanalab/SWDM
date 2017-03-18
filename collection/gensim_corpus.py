import os

from gensim.corpora import TextCorpus

from index.index import Index

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


class GensimCorpus(TextCorpus):
    def         __init__(self, parameters):
        self.parameters = parameters
        self.index_ = Index(self.parameters)
        self.store_collection_if_not_exists()
        input_ = self.parameters.params["lda"]["file_name"]
        super().__init__(input_)

    def read_collection(self):
        collection_lines = ""
        for i in range(1, self.index_.total_count() + 1):
            l = self.index_.obtain_terms_of_a_document(i)[1]
            l = [w for w in l if w.isalpha() and len(w) > 2]
            collection_lines += ' '.join(l) + '\n'
        return collection_lines

    def store_collection_if_not_exists(self):
        if not os.path.exists(self.parameters.params["lda"]["file_name"]):
            collection_lines = self.read_collection()
            with open(self.parameters.params["lda"]["file_name"], "w") as f:
                f.write(collection_lines)
