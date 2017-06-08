import enchant
from nltk import word_tokenize
from nltk.corpus import stopwords

from index.index import Index

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 6 / 8 / 17


class Original:
    def __init__(self, parameters):
        self.parameters = parameters
        self.enchant_dict = enchant.Dict("en_US")
        self.stopwords = stopwords.words('english')
        self.index_ = Index(self.parameters)

    def check_if_unigram_should_be_added(self, unigram):

        unigram = unigram.lower()

        if not unigram.isalpha():
            return False
        if unigram in self.stopwords:
            return False
        if not self.index_.check_if_exists_in_index(unigram):
            return False
        # uncomment to include a dictionary
        # if not self.enchant_dict.check(unigram):
        #     print("WARNING: \"", unigram, "\" doesn't exist in dictionary.", file=sys.stderr, end=" ")
        #     return False
        return True

    def find_unigrams(self, text):
        for unigram in word_tokenize(text):
            if self.check_if_unigram_should_be_added(unigram):
                yield [(unigram, 1)]
