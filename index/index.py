import math

import pyndri
import sys
from bs4 import BeautifulSoup


class Index:
    def __init__(self, parameters):
        self.repo_dir = parameters.params['repo_dir']
        self.index = pyndri.Index(self.repo_dir)
        self.query_env = None

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

    @staticmethod
    def gen_expression_query(term, operator, window_size):
        return operator + str(window_size) + "(" + term + ")"

    def expression_count(self, term, operator, window_size):
        query = self.gen_expression_query(term, operator, window_size)
        return self.index.expression_count(query)

    def expression_list(self, term, operator, window_size):
        query = self.gen_expression_query(term, operator, window_size)
        return self.index.expression_list(query)

    def expression_document_count(self, term, operator, window_size):
        query = self.gen_expression_query(term, operator, window_size)
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

    def init_query_env(self, rules=('method:linear,collectionLambda:0.4,documentLambda:0.2',)):
        self.query_env = pyndri.QueryEnvironment(
            self.index,
            rules=rules)

    def run_query(self, query):
        return self.query_env.query(query)

    def get_ext_document_id(self, doc_id):
        if not isinstance(doc_id, int):
            raise TypeError("doc_id is \"" + str(doc_id) + "\" which is not an integer")
        return self.index.ext_document_id(doc_id)

    def run_query_doc_names(self, query):
        runs = self.run_query(query)
        return [self.get_ext_document_id(i) for i, j in runs]

    def expression_list_in_top_docs(self, term, operator, window_size, n_top_docs):
        query = self.gen_expression_query(term, operator, window_size)
        runs = self.run_query_doc_names(query)
        del runs[n_top_docs:]
        runs = set(runs)
        expression_list = self.index.expression_list(query)
        return {k: v for k, v in expression_list.items() if k in runs}


if __name__ == '__main__':
    pass
