import gensim


class Word2vec:
    def __init__(self):
        self.model = None
        pass

    def pre_trained_google_news_300_model(self):
        self.model = gensim.models.Word2Vec.load_word2vec_format('/scratch/data/GoogleNews-vectors-negative300.bin',
                                                                 binary=True)

    def train_model_from_sentences(self, sentences):
        self.model = gensim.models.Word2Vec(sentences, min_count=1)

    @staticmethod
    def sentences(fname):
        for line in open(fname):
            yield line.split()

    def gen_similar_words(self, unigram, topn):
        res = self.model.most_similar([unigram], [], topn)
        return res


if __name__ == '__main__':
    pass
