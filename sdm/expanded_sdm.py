from sdm.weights.ordered_bigram_weights import OrderedBigramWeights
from sdm.weights.unigram_weights import UnigramWeights
from sdm.weights.unordered_bigram_weights import UnorderedBigramWeights


class ExpandedSdm:
    def __init__(self, parameters):
        self.parameters = parameters
        self.unigram_weights = UnigramWeights(self.parameters)
        self.unordered_bigram_weights = UnorderedBigramWeights(self.parameters)
        self.ordered_bigram_weights = OrderedBigramWeights(self.parameters)

    def compute_weight_sdm_unigrams(self, similar_unigram, unigram_nearest_neighbor):
        term_dependent_feature_parameters = {
            "unigram_nearest_neighbor": unigram_nearest_neighbor,
        }
        return self.unigram_weights.compute_weight(similar_unigram, term_dependent_feature_parameters)

    def compute_weight_sdm_bigrams(self, term, unigram_nearest_neighbor_1, unigram_nearest_neighbor_2, operator):
        term_dependent_feature_parameters = {
            "unigram_nearest_neighbor_1": unigram_nearest_neighbor_1,
            "unigram_nearest_neighbor_2": unigram_nearest_neighbor_2
        }
        if operator == "#uw":
            return self.unordered_bigram_weights.compute_weight(term, term_dependent_feature_parameters)
        elif operator == "#od":
            return self.ordered_bigram_weights.compute_weight(term, term_dependent_feature_parameters)

    def gen_sdm_field_1_text(self, unigrams_in_embedding_space, operator):
        if operator == "#combine":
            return self.gen_sdm_unigrams_field_1_text(unigrams_in_embedding_space)
        elif operator in {"#uw", "#od"}:
            return self.gen_sdm_bigrams_field_1_text(unigrams_in_embedding_space, operator)

    def gen_sdm_bigrams_field_1_text(self, unigrams_in_embedding_space, operator):
        sdm_bigrams_field_text = "#weight(\n"
        for i in range(0, len(unigrams_in_embedding_space) - 1):
            for similar_unigram_1 in unigrams_in_embedding_space[i]:
                for similar_unigram_2 in unigrams_in_embedding_space[i + 1]:
                    bigram = similar_unigram_1[0] + " " + similar_unigram_2[0]
                    weight = self.compute_weight_sdm_bigrams(bigram, unigrams_in_embedding_space[i],
                                                             unigrams_in_embedding_space[i + 1],
                                                             operator)
                    weight = '{0:.10f}'.format(weight)
                    if float(weight) <= 0:
                        continue
                    operator_s = operator + str(self.parameters.params["window_size"][operator])
                    sdm_bigrams_field_text += weight + operator_s + "(" + bigram + ")\n"

        sdm_bigrams_field_text += ")\n"
        return sdm_bigrams_field_text

    def gen_sdm_unigrams_field_1_text(self, unigrams_in_embedding_space):
        sdm_unigrams_field_text = "#weight(\n"
        operator = "#combine"
        for unigram_nearest_neighbor in unigrams_in_embedding_space:
            for similar_unigram in unigram_nearest_neighbor:
                weight = self.compute_weight_sdm_unigrams(similar_unigram[0], unigram_nearest_neighbor)
                weight = '{0:.10f}'.format(weight)
                if float(weight) <= 0:
                    continue
                sdm_unigrams_field_text += weight + operator + "(" + similar_unigram[0] + ")\n"
        sdm_unigrams_field_text += ")\n"
        return sdm_unigrams_field_text
