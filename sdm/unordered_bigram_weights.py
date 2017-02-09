import features.features


class UnorderedBigramWeights:
    def __init__(self):
        self.features = features.features.Features
        self.feature_names = {
            "uw_expression_norm_count",
            "uw_expression_norm_document_count",
            "bigrams_cosine_similarity_with_orig",
        }
