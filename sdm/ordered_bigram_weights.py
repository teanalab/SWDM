import features.features


class OrderedBigramWeights:
    def __init__(self):
        self.features = features.features.Features
        self.feature_names = {
            "od_expression_norm_count",
            "od_expression_norm_document_count",
            "bigrams_cosine_similarity_with_orig",
        }
