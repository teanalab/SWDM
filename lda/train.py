import os

import gensim
import sys

from collection.gensim_corpus import GensimCorpus


class Train:
    def __init__(self, parameters):
        self.corpus = None
        self.lda = None
        self.parameters = parameters
        pass

    def add_corpus(self):
        if not os.path.exists(self.parameters.params["lda"]["model"]):
            self.corpus = GensimCorpus(self.parameters)

    def run(self):
        if not os.path.exists(self.parameters.params["lda"]["model"]):
            print("training model...", file=sys.stderr)
            self.lda = gensim.models.ldamodel.LdaModel(corpus=self.corpus, id2word=self.corpus.dictionary,
                                                       num_topics=self.parameters.params["lda"]["num_topics"],
                                                       update_every=1, chunksize=10000, passes=1)
            print("write model to:", self.parameters.params["lda"]["model"], file=sys.stderr)
            self.lda.save(self.parameters.params["lda"]["model"])
        else:
            print("read model from:", self.parameters.params["lda"]["model"], file=sys.stderr)
            id2word = gensim.corpora.Dictionary.load(self.parameters.params["lda"]["model"]+".id2word")
            print("id2word:", id2word, file=sys.stderr)
            self.lda = gensim.models.ldamodel.LdaModel(id2word=id2word)
            print("loading model...", file=sys.stderr)
            self.lda.load(self.parameters.params["lda"]["model"])
            print("model loaded.", file=sys.stderr)
