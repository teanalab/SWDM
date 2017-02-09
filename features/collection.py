import index.index


class Collection:
    def __init__(self, repo_dir):
        self.index = index.index.Index(repo_dir)
        self.repo_dir = repo_dir
        pass

    def uw_expression_count(self, term, feature_parameters):
        return self.index.uw_expression_count(term, feature_parameters["window_size"])

    def od_expression_count(self, term, feature_parameters):
        return self.index.od_expression_count(term, feature_parameters["window_size"])

    def uw_document_expression_count(self, term, feature_parameters):
        return self.index.uw_document_expression_count(term, feature_parameters["window_size"])

    def od_document_expression_count(self, term, feature_parameters):
        return self.index.od_document_expression_count(term, feature_parameters["window_size"])

    def term_count(self, term, feature_parameters):
        del feature_parameters
        return self.index.term_count(term)

    def document_count(self, term, feature_parameters):
        del feature_parameters
        return self.index.document_count(term)