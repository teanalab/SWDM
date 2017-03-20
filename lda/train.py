import os

import gensim

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
            self.lda = gensim.models.ldamodel.LdaModel(corpus=self.corpus, id2word=self.corpus.dictionary,
                                                       num_topics=self.parameters.params["lda"]["num_topics"],
                                                       update_every=1, chunksize=10000, passes=1)
            self.lda.save(self.parameters.params["lda"]["model"])
        else:
            self.lda = gensim.models.ldamodel.LdaModel.load(self.parameters.params["lda"]["model"])
