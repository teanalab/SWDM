import nltk

from index.index import Index


class SimpleDocument:
    def __init__(self, parameters):
        self.index_ = Index(parameters)
        self.stop_words = set(nltk.corpus.stopwords.words('english'))

    def replace_stemmed_similar_words_list(self, l):
        i = 0
        while i < len(l):
            j = i + 1
            while j < len(l):
                if self.index_.check_if_have_same_stem(l[i], l[j]):
                    l[j] = l[i]
                j += 1
            i += 1
        return l

    def remove_non_existent_words_in_repo(self, l):
        return [w for w in l if self.index_.check_if_exists_in_index(w)]

    def get_words(self, doc_file_name):
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        with open(doc_file_name, 'r') as f:
            doc_words = tokenizer.tokenize(f.read())
        doc_words = [w.lower() for w in doc_words]
        doc_words = [w for w in doc_words if w not in self.stop_words]
        doc_words = self.remove_non_existent_words_in_repo(doc_words)
        doc_words = self.replace_stemmed_similar_words_list(doc_words)
        doc_words = [w for w in doc_words if w.isalpha()]
        doc_words = [w for w in doc_words if len(w) > 2]
        return doc_words
