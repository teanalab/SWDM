import pyndri


class Index:
    def __init__(self, repo_dir='/scratch/index/indri_5_7/robust'):
        self.index = pyndri.Index(repo_dir)
        pass

    def expression_count(self, term, operator, window_size):
        query = operator + str(window_size) + "(" + term + ")"
        return self.index.expression_count(query)

    def document_expression_count(self, term, operator, window_size):
        query = operator + str(window_size) + "(" + term + ")"
        return self.index.document_expression_count(query)

    def uw_expression_count(self, term, window_size):
        return self.expression_count(term, "#uw", window_size)

    def od_expression_count(self, term, window_size):
        return self.expression_count(term, "#od", window_size)

    def uw_document_expression_count(self, term, window_size):
        return self.document_expression_count(term, "#uw", window_size)

    def od_document_expression_count(self, term, window_size):
        return self.document_expression_count(term, "#od", window_size)

    def term_count(self, term):
        return self.index.term_count(term)

    def document_count(self, term):
        return self.index.document_count(term)

if __name__ == '__main__':
    pass
