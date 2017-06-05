from math import log

import index.index


class Topdocs:
    def __init__(self, parameters):
        self.index = index.index.Index(parameters)
        self.index.init_query_env()
        self.runs = None

    def init_top_docs_run_query(self, query):
        self.runs = self.index.run_query_doc_names(query)

    def uw_expression_count(self, term, feature_parameters):
        return sum(self.index.expression_list_in_top_docs(term, "#uw", feature_parameters["window_size"],
                                                          feature_parameters["n_top_docs"], self.runs).value())

    def od_expression_count(self, term, feature_parameters):
        return sum(self.index.expression_list_in_top_docs(term, "#od", feature_parameters["window_size"],
                                                          feature_parameters["n_top_docs"], self.runs).value())

    def uw_expression_document_count(self, term, feature_parameters):
        return len(self.index.expression_list_in_top_docs(term, "#uw", feature_parameters["window_size"],
                                                          feature_parameters["n_top_docs"], self.runs))

    def od_expression_document_count(self, term, feature_parameters):
        return len(self.index.expression_list_in_top_docs(term, "#od", feature_parameters["window_size"],
                                                          feature_parameters["n_top_docs"], self.runs))

    def term_count(self, term, feature_parameters):
        return len(self.index.expression_list_in_top_docs_exp(expression=term, runs=self.runs,
                                                              n_top_docs=feature_parameters["n_top_docs"]))

    def document_count(self, term, feature_parameters):
        return sum(self.index.expression_list_in_top_docs_exp(expression=term, runs=self.runs,
                                                              n_top_docs=feature_parameters["n_top_docs"]).values)

    def uw_expression_norm_count(self, term, feature_parameters):
        total_terms = self.index.document_length_docs_names(self.runs[:feature_parameters["n_top_docs"]])
        return -log((self.uw_expression_count(term, feature_parameters) + 1) / total_terms)

    def od_expression_norm_count(self, term, feature_parameters):
        total_terms = self.index.document_length_docs_names(self.runs[:feature_parameters["n_top_docs"]])
        return -log((self.od_expression_count(term, feature_parameters) + 1) / total_terms)

    def uw_expression_norm_document_count(self, term, feature_parameters):
        return -log((self.uw_expression_document_count(term, feature_parameters) + 1) / feature_parameters["n_top_docs"])

    def od_expression_norm_document_count(self, term, feature_parameters):
        return -log((self.od_expression_document_count(term, feature_parameters) + 1) / feature_parameters["n_top_docs"])

    def norm_term_count(self, term, feature_parameters):
        total_terms = self.index.document_length_docs_names(self.runs[:feature_parameters["n_top_docs"]])
        return -log((self.term_count(term, feature_parameters) + 1) / total_terms)

    def norm_document_count(self, term, feature_parameters):
        return -log((self.document_count(term, feature_parameters) + 1) / feature_parameters["n_top_docs"])
