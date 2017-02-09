class ExpandedSdm:
    @staticmethod
    def compute_weight_sdm_unigrams(similar_unigram, unigram):
        weight = 0
        return weight

    @staticmethod
    def compute_weight_sdm_bigrams(similar_unigram_1, unigram_1, similar_unigram_2, unigram_2):
        weight = 0
        return weight

    def gen_sdm_bigrams_field_1_text(self, unigrams_in_embedding_space, operator):
        sdm_bigrams_field_text = "#weight(\n"
        for i in range(0, len(unigrams_in_embedding_space) - 1):
            for similar_unigram_1 in unigrams_in_embedding_space[i]:
                for similar_unigram_2 in unigrams_in_embedding_space[i + 1]:
                    weight = self.compute_weight_sdm_bigrams(similar_unigram_1, unigrams_in_embedding_space[i],
                                                             similar_unigram_2, unigrams_in_embedding_space[i + 1])
                    bigram = similar_unigram_1[0] + " " + similar_unigram_2[0]
                    sdm_bigrams_field_text += str(weight) + \
                                              " " + operator + "(" + bigram + ")\n"

        sdm_bigrams_field_text += ")\n"
        return sdm_bigrams_field_text

    def gen_sdm_unigrams_field_1_text(self, unigrams_in_embedding_space, operator):
        sdm_unigrams_field_text = "#weight(\n"
        for unigram_nearest_neighbor in unigrams_in_embedding_space:
            for similar_unigram in unigram_nearest_neighbor:
                sdm_unigrams_field_text += str(self.compute_weight_sdm_unigrams(similar_unigram, unigram_nearest_neighbor)) + \
                                           " " + operator + "(" + similar_unigram[0] + ")\n"
        sdm_unigrams_field_text += ")\n"
        return sdm_unigrams_field_text
