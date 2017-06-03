import math

import pyndri
from bs4 import BeautifulSoup


class Index:
    def __init__(self, parameters):
        self.repo_dir = parameters.params['repo_dir']
        self.index = pyndri.Index(self.repo_dir)

    def check_if_exists_in_index(self, unigram):
        return self.index.process_term(unigram) != "" and self.document_count(unigram) > 0

    def check_if_have_same_stem(self, unigram1, unigram2):
        unigram1_ = self.index.process_term(unigram1)
        unigram2_ = self.index.process_term(unigram2)
        if unigram1_ == "":
            raise LookupError("unigram \"" + unigram1 + "\" not exist. Probably was a stopword in indexing.")
        if unigram2_ == "":
            raise LookupError("unigram \"" + unigram2 + "\" not exist. Probably was a stopword in indexing.")
        return unigram1_ == unigram2_

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
        return math.log(self.index.total_count() / (self.document_count(term)))

    @staticmethod
    def tf(term, doc_terms):
        max_f = max([doc_terms.count(term_) for term_ in doc_terms])
        return 0.5 + 0.5 * doc_terms.count(term) / max_f

    def tfidf(self, term, doc_terms):
        if not self.check_if_exists_in_index(term):
            raise LookupError("unigram \"" + term + "\" not exist. Probably was a stopword in indexing.")
        return self.tf(term, doc_terms) * self.idf(term)

    def obtain_text_of_a_document(self, doc_id):
        text = self.index.document_text(doc_id)
        return BeautifulSoup(text, "lxml").find("text").text

    def obtain_term_ids_of_a_document(self, doc_id):
        return self.index.document_term_ids(doc_id)

    def obtain_terms_of_a_document(self, doc_id):
        return self.index.document_terms(doc_id)

    def term(self, term_id):
        return self.index.term(term_id)

if __name__ == '__main__':
    pass
