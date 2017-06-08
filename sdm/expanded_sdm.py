from sdm.weights.ordered_bigram_weights import OrderedBigramWeights
from sdm.weights.unigram_weights import UnigramWeights
from sdm.weights.unordered_bigram_weights import UnorderedBigramWeights

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class ExpandedSdm:
    def __init__(self, parameters):
        self.parameters = parameters
        self.unigram_weights = UnigramWeights(self.parameters)
        self.unordered_bigram_weights = UnorderedBigramWeights(self.parameters)
        self.ordered_bigram_weights = OrderedBigramWeights(self.parameters)

    def init_top_docs_run_query(self, query):
        self.unigram_weights.init_top_docs_run_query(query)
        self.unordered_bigram_weights.init_top_docs_run_query(query)
        self.ordered_bigram_weights.init_top_docs_run_query(query)

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

    def gen_sdm_field_1_text(self, all_unigrams, operator):
        sdm_a_field_text = "#weight(\n"
        for concept_type, unigrams in all_unigrams.items():
            type_weight = self.parameters.params["type_weights"][concept_type]
            if operator == "#combine":
                sdm_a_field_text += str(type_weight) + self.gen_sdm_unigrams_field_1_text(unigrams)
            elif operator in {"#uw", "#od"}:
                sdm_a_field_text += str(type_weight) + self.gen_sdm_bigrams_field_1_text(unigrams, operator)
            sdm_a_field_text += ")\n"
        return sdm_a_field_text

    def gen_sdm_bigrams_field_1_text(self, unigrams_in_embedding_space, operator):
        sdm_bigrams_field_text = "#weight(\n"
        for i in range(0, len(unigrams_in_embedding_space) - 1):
            for similar_unigram_1 in unigrams_in_embedding_space[i]:
                for similar_unigram_2 in unigrams_in_embedding_space[i + 1]:
                    if similar_unigram_1[0].strip() == "" or similar_unigram_2[0].strip() == "":
                        continue
                    bigram = similar_unigram_1[0] + " " + similar_unigram_2[0]
                    weight = self.compute_weight_sdm_bigrams(bigram, unigrams_in_embedding_space[i],
                                                             unigrams_in_embedding_space[i + 1],
                                                             operator)
                    weight = '{0:.20f}'.format(weight)
                    if float(weight) <= 0 or len(bigram) == 0:
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
                if similar_unigram[0].strip() == "":
                    continue
                weight = self.compute_weight_sdm_unigrams(similar_unigram[0], unigram_nearest_neighbor)
                weight = '{0:.20f}'.format(weight)
                if float(weight) <= 0 or len(similar_unigram[0]) == 0:
                    continue
                sdm_unigrams_field_text += weight + operator + "(" + similar_unigram[0] + ")\n"
        sdm_unigrams_field_text += ")\n"
        return sdm_unigrams_field_text
