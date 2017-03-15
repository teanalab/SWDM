import math
import pyndri
import sys


class Index:
    def __init__(self, parameters):
        self.repo_dir = parameters.params['repo_dir']
        self.index = pyndri.Index(self.repo_dir)

    def check_if_have_same_stem(self, unigram1, unigram2):
        return self.index.process_term(unigram1) == self.index.process_term(unigram2)

    def expression_count(self, term, operator, window_size):
        query = operator + str(window_size) + "(" + term + ")"
        return self.index.expression_count(query)

    def expression_document_count(self, term, operator, window_size):
        query = operator + str(window_size) + "(" + term + ")"
        return self.index.document_expression_count(query)

    def uw_expression_count(self, term, window_size):
        return self.expression_count(term, "#uw", window_size)

    def od_expression_count(self, term, window_size):
        return self.expression_count(term, "#od", window_size)

    def uw_expression_document_count(self, term, window_size):
        return self.expression_document_count(term, "#uw", window_size)

    def od_expression_document_count(self, term, window_size):
        return self.expression_document_count(term, "#od", window_size)

    def term_count(self, term):
        return self.index.term_count(term)

    def document_count(self, term):
        return self.index.document_count(term)

    def total_count(self):
        return self.index.total_count()

    def total_terms(self):
        return self.index.total_terms()

    def idf(self, term):
        return math.log(self.index.total_count()/(1+self.document_count(term)))

    def tf(self, term, max_tf):
        return 0.5 + 0.5 * self.term_count(term) / max_tf

    def tfidf_fast(self, term, max_tf):
        print(term, file=sys.stderr)
        print(self.tf(term, max_tf), file=sys.stderr)
        print(self.idf(term), file=sys.stderr)
        return self.tf(term, max_tf) * self.idf(term)

    def tfidf(self, term, doc_terms):
        max_tf = max([self.term_count(term_) for term_ in doc_terms])
        return self.tfidf_fast(term, max_tf)

if __name__ == '__main__':
    pass
