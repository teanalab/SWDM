import gensim


class Word2vec:
    def __init__(self):
        self.model = gensim.models.Word2Vec.load_word2vec_format('/scratch/data/GoogleNews-vectors-negative300.bin',
                                                                 binary=True)

    def run(self, unigram, topn):
        res = self.model.most_similar([unigram], [], topn)
        return res

if __name__ == '__main__':
    pass
