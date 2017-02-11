from sdm.weights.ordered_bigram_weights import OrderedBigramWeights
from sdm.weights.unigram_weights import UnigramWeights
from sdm.weights.unordered_bigram_weights import UnorderedBigramWeights


class ExpandedSdm:
    def __init__(self, repo_dir):
        self.repo_dir = repo_dir

    def compute_weight_sdm_unigrams(self, similar_unigram, unigram_nearest_neighbor):
        unigram_weights = UnigramWeights(self.repo_dir)
        term_dependent_feature_parameters = {
            "unigram_nearest_neighbor": unigram_nearest_neighbor,
        }
        return unigram_weights.compute_weight(similar_unigram, term_dependent_feature_parameters)

    def compute_weight_sdm_bigrams(self, similar_unigram_1, unigram_nearest_neighbor_1,
                                   similar_unigram_2, unigram_nearest_neighbor_2, operator):
        term = similar_unigram_1 + " " + similar_unigram_2
        term_dependent_feature_parameters = {
            "unigram_nearest_neighbor_1": unigram_nearest_neighbor_1,
            "unigram_nearest_neighbor_2": unigram_nearest_neighbor_2
        }
        if operator == "#uw":
            unordered_bigram_weights = UnorderedBigramWeights(self.repo_dir)
            return unordered_bigram_weights.compute_weight(term, term_dependent_feature_parameters)
        elif operator == "#od":
            ordered_bigram_weights = OrderedBigramWeights(self.repo_dir)
            return ordered_bigram_weights.compute_weight(term, term_dependent_feature_parameters)

    def gen_sdm_field_1_text(self, unigrams_in_embedding_space, operator):
        if operator is "#combine":
            return self.gen_sdm_unigrams_field_1_text(unigrams_in_embedding_space)
        elif operator in {"#uw", "#od"}:
            return self.gen_sdm_bigrams_field_1_text(unigrams_in_embedding_space, operator)

    def gen_sdm_bigrams_field_1_text(self, unigrams_in_embedding_space, operator):
        sdm_bigrams_field_text = "#weight(\n"
        for i in range(0, len(unigrams_in_embedding_space) - 1):
            for similar_unigram_1 in unigrams_in_embedding_space[i]:
                for similar_unigram_2 in unigrams_in_embedding_space[i + 1]:
                    weight = self.compute_weight_sdm_bigrams(similar_unigram_1, unigrams_in_embedding_space[i],
                                                             similar_unigram_2, unigrams_in_embedding_space[i + 1],
                                                             operator)
                    bigram = similar_unigram_1[0] + " " + similar_unigram_2[0]
                    sdm_bigrams_field_text += str(weight) + operator + "(" + bigram + ")\n"

        sdm_bigrams_field_text += ")\n"
        return sdm_bigrams_field_text

    def gen_sdm_unigrams_field_1_text(self, unigrams_in_embedding_space):
        sdm_unigrams_field_text = "#weight(\n"
        operator = "#combine"
        for unigram_nearest_neighbor in unigrams_in_embedding_space:
            for similar_unigram in unigram_nearest_neighbor:
                weight = self.compute_weight_sdm_unigrams(similar_unigram, unigram_nearest_neighbor)
                sdm_unigrams_field_text += str(weight) + operator + "(" + similar_unigram[0] + ")\n"
        sdm_unigrams_field_text += ")\n"
        return sdm_unigrams_field_text
