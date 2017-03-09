import sys


class Neighborhood:
    def __init__(self, word2vec_model):
        self.word2vec_model = word2vec_model

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

    def find_significant_neighbors(self, other_unigrams, min_distance, neighbor_size):
        significant_neighbors = []
        for other_unigram in other_unigrams:
            if other_unigram in self.word2vec_model.wv.vocab:
                neighbor = self.find_nearest_neighbor_in_a_list(other_unigram, other_unigrams, min_distance,
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
                print("i =", i, file=sys.stderr)
                print("j =", j, file=sys.stderr)
                print("neighbors =", neighbors, file=sys.stderr)
                print("neighbors[j] =", neighbors[j], file=sys.stderr)
                print("before merged_neighbors[i] =", merged_neighbors[i], file=sys.stderr)
                neighbor_intersection = merged_neighbors[i].intersection(neighbors[j])
                if len(neighbor_intersection) >= minimum_merge_intersection:
                    merged_neighbors[i] = set(merged_neighbors[i]).union(neighbors[j])
                    del neighbors[j]
                else:
                    j += 1
                print("after merged_neighbors[i] =", merged_neighbors[i], file=sys.stderr)
            i += 1
        return merged_neighbors
