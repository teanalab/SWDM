import nltk

from index.index import Index


class Neighborhood:
    def __init__(self, word2vec_model, parameters):
        self.word2vec_model = word2vec_model
        self.index_ = Index(parameters)

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

    @staticmethod
    def remove_stopwords_neighbors(neighbors, max_stop_words):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        i = 0
        while i < len(neighbors):
            neighbor_stop_words_intersection = set(neighbors[i]).intersection(set(stop_words))
            if len(neighbor_stop_words_intersection) >= max_stop_words:
                del neighbors[i]
            else:
                for a in neighbors[i].copy():
                    if a in stop_words:
                        neighbors[i].remove(a)
                i += 1
        return neighbors

    def remove_stemmed_similar_words(self, neighbors):
        for k in range(len(neighbors)):
            neighbor_ = list(neighbors[k])
            i = 0
            while i < len(neighbor_):
                j = i + 1
                while j < len(neighbor_):
                    if self.index_.check_if_have_same_stem(neighbor_[i], neighbor_[j]):
                        del neighbor_[j]
                    else:
                        j += 1
                i += 1
            neighbors[k] = set(neighbor_)
        return neighbors

    def find_significant_pruned_neighbors(self, doc_words, min_distance, neighbor_size, minimum_merge_intersection,
                                          max_stop_words):
        significant_neighbors = \
            self.find_significant_merged_neighbors(doc_words, min_distance, neighbor_size, minimum_merge_intersection)
        significant_neighbors = self.remove_stopwords_neighbors(significant_neighbors, max_stop_words)
        significant_neighbors = self.remove_stemmed_similar_words(significant_neighbors)
        return significant_neighbors
