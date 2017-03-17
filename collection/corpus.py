from gensim.corpora import TextCorpus
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


class Corpus(TextCorpus):
    def __init__(self):
        super().__init__()

    def getstream(self):
        return StringIO("some initial text data")

    def get_texts(self):
        super().get_texts()
