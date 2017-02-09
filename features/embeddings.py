

class Embeddings:

    def __init__(self):
        pass

    @staticmethod
    def cosine_similarity_with_orig(term, feature_parameters):
        unigram_nearest_neighbor = feature_parameters['unigram_nearest_neighbor']
        return [item for item in unigram_nearest_neighbor if item[0] == term][0][1]
