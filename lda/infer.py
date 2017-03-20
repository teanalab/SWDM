import os
import sys

import nltk

from collection.gensim_corpus import GensimCorpus


class Infer:
    def __init__(self, parameters):
        self.parameters = parameters
        self.corpus = None
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    def add_corpus(self):
        self.corpus = GensimCorpus(self.parameters)

    def get_words(self, doc_text):
        doc_words = self.tokenizer.tokenize(doc_text.lower())
        doc_words = [w for w in doc_words if w.isalpha() and len(w) > 2 and w not in self.stop_words]
        return doc_words

    def infer_topics(self, doc_text, lda):
        doc_words = self.get_words(doc_text)
        bow = self.corpus.dictionary.doc2bow(doc_words)
        sparse_topics = lda.get_document_topics(bow)
        print(sparse_topics, file=sys.stderr)
        lda.
        # nparray = gensim.matutils.sparse2full(sparse_topics, num_topics)
        # return nparray.tolist()
        return sparse_topics
