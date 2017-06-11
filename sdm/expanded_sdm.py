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

    def compute_weight_sdm_grams(self, gram_pair, operator):
        if operator == "#combine":
            return self.unigram_weights.compute_weight(gram_pair)
        elif operator == "#uw":
            return self.unordered_bigram_weights.compute_weight(gram_pair)
        elif operator == "#od":
            return self.ordered_bigram_weights.compute_weight(gram_pair)

    def gen_sdm_field_1_text(self, all_grams, operator):
        sdm_a_field_text = ""
        for concept_type, grams_1_type in all_grams.items():
            type_weight = self.parameters.params["type_weights"][concept_type]
            type_string = self.gen_sdm_grams_field_1_text(grams_1_type, operator)
            if type_string is not None and type_string.strip() is not "":
                sdm_a_field_text += " " * 4 + str(type_weight) + type_string
        if len(sdm_a_field_text.strip()) > 2:
            sdm_a_field_text = "#weight(\n" + sdm_a_field_text + " " * 2 + ")\n"
            return sdm_a_field_text
        else:
            return ""

    def gen_sdm_grams_field_1_text(self, grams, operator):
        sdm_grams_field_text = ""
        for neighbor in grams:
            gram_orig = neighbor[0]
            for gram_exp in neighbor[1]:
                gram_pair = (gram_orig, gram_exp)
                weight = self.compute_weight_sdm_grams(gram_pair, operator)
                weight = '{0:.20f}'.format(weight)
                if float(weight) <= 0 or len(gram_exp[0]) == 0:
                    continue
                if isinstance(gram_exp[0], str):
                    gram_exp = gram_exp[0]
                elif isinstance(gram_exp[0], tuple):
                    gram_exp = ' '.join([gram_exp[0][0], gram_exp[1][0]])
                sdm_grams_field_text += " " * 6 + weight + operator + "(" + gram_exp + ")\n"
        if len(sdm_grams_field_text.strip()) > 2:
            sdm_grams_field_text = "#weight(\n" + sdm_grams_field_text + " " * 4 + ")\n"
            return sdm_grams_field_text
        else:
            return ""
