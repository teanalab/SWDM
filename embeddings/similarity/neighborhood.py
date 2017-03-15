import nltk
import sys

from index.index import Index


class Neighborhood:
    def __init__(self, word2vec_model, parameters):
        self.word2vec_model = word2vec_model
        self.index_ = Index(parameters)
        self.stop_words = set(nltk.corpus.stopwords.words('english'))

    def find_nearest_neighbor_in_a_list(self, unigram, other_unigrams, min_distance, neighbor_size):
        neighbor = []
        if unigram in self.word2vec_model.wv.vocab:
            for other_unigram in other_unigrams:
                if len(neighbor) > neighbor_size:
                    break
                if other_unigram is not unigram and other_unigram not in neighbor and \
                                other_unigram in self.word2vec_model.wv.vocab:
                    sim = self.word2vec_model.similarity(unigram, other_unigram)
                    if sim > min_distance:
                        neighbor += [other_unigram]
        return neighbor

    def find_significant_neighbors(self, doc_words, min_distance, neighbor_size):
        significant_neighbors = []
        for other_unigram in doc_words:
            if other_unigram in self.word2vec_model.wv.vocab:
                neighbor = self.find_nearest_neighbor_in_a_list(other_unigram, doc_words, min_distance,
                                                                neighbor_size)
                if len(neighbor) == neighbor_size:
                    significant_neighbors += [neighbor]
        significant_neighbors = [list(x) for x in set(tuple(x) for x in significant_neighbors)]
        return significant_neighbors

    @staticmethod
    def merge_close_neighbors(neighbors, minimum_merge_intersection):
        merged_neighbors = []
        i = 0
        while i < len(neighbors):
            merged_neighbors += [set(neighbors[i])]
            j = i + 1
            while j < len(neighbors):
                neighbor_intersection = merged_neighbors[i].intersection(neighbors[j])
                if len(neighbor_intersection) >= minimum_merge_intersection:
                    merged_neighbors[i] = set(merged_neighbors[i]).union(neighbors[j])
                    del neighbors[j]
                else:
                    j += 1
            i += 1
        return merged_neighbors

    def find_significant_merged_neighbors(self, doc_words, min_distance, neighbor_size, minimum_merge_intersection):
        significant_neighbors = self.find_significant_neighbors(doc_words, min_distance, neighbor_size)
        significant_merged_neighbors = self.merge_close_neighbors(significant_neighbors, minimum_merge_intersection)
        return significant_merged_neighbors

    def remove_stopwords_neighbors(self, neighbors, max_stop_words):
        i = 0
        while i < len(neighbors):
            neighbor_stop_words_intersection = set(neighbors[i]).intersection(set(self.stop_words))
            if len(neighbor_stop_words_intersection) >= max_stop_words:
                del neighbors[i]
            else:
                for a in neighbors[i].copy():
                    if a in self.stop_words:
                        neighbors[i].remove(a)
                i += 1
        return neighbors

    def remove_stemmed_similar_words_neighbors(self, neighbors):
        for k in range(len(neighbors)):
            neighbor_ = list(neighbors[k])
            neighbors[k] = set(self.remove_stemmed_similar_words_list(neighbor_))
        return neighbors

    def remove_stemmed_similar_words_list(self, l):
        i = 0
        while i < len(l):
            j = i + 1
            while j < len(l):
                if self.index_.check_if_have_same_stem(l[i], l[j]):
                    del l[j]
                else:
                    j += 1
            i += 1
        return l

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

    def find_significant_pruned_neighbors(self, doc_words, min_distance, neighbor_size, minimum_merge_intersection,
                                          max_stop_words):
        doc_words = list(set(doc_words))
        significant_neighbors = \
            self.find_significant_merged_neighbors(doc_words, min_distance, neighbor_size, minimum_merge_intersection)
        significant_neighbors = self.remove_stopwords_neighbors(significant_neighbors, max_stop_words)
        significant_neighbors = self.remove_stemmed_similar_words_neighbors(significant_neighbors)
        return significant_neighbors

    def get_words(self, doc_file_name):
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        with open(doc_file_name, 'r') as f:
            doc_words = tokenizer.tokenize(f.read())
        doc_words = [w.lower() for w in doc_words]
        doc_words = [w for w in doc_words if w not in self.stop_words]
        doc_words = self.replace_stemmed_similar_words_list(doc_words)
        doc_words = [w for w in doc_words if w.isalpha()]
        doc_words = [w for w in doc_words if len(w) > 2]
        return doc_words

    def find_significant_pruned_neighbors_in_doc(self, doc_file_name, min_distance, neighbor_size,
                                                 minimum_merge_intersection,
                                                 max_stop_words):

        doc_words = self.get_words(doc_file_name)

        significant_neighbors = self.find_significant_pruned_neighbors(doc_words, min_distance, neighbor_size,
                                                                       minimum_merge_intersection, max_stop_words)
        return significant_neighbors
