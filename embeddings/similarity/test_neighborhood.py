import sys
from unittest import TestCase

from embeddings.similarity.neighborhood import Neighborhood
from embeddings.word2vec import Word2vec
from parameters.parameters import Parameters


class TestNeighborhood(TestCase):
    def setUp(self):
        self.word2vec = Word2vec()
        self.parameters = Parameters()
        self.parameters.params["repo_dir"] = '../../index/test_files/index'

        self.other_unigrams = ['Modern', 'humans', '(Homo', 'sapiens,', 'primarily', 'ssp.', 'Homo', 'sapiens',
                               'sapiens)', 'are', 'the', 'only', 'extant', 'members', 'of', 'Hominina', 'tribe', '(or',
                               'human', 'tribe),', 'a', 'branch', 'of', 'the', 'tribe', 'Hominini', 'belonging', 'to',
                               'the', 'family', 'of', 'great', 'apes.', 'They', 'are', 'characterized', 'by', 'erect',
                               'posture', 'and', 'bipedal', 'locomotion;', 'manual', 'dexterity', 'and', 'increased',
                               'tool', 'use,', 'compared', 'to', 'other', 'animals;', 'and', 'a', 'general', 'trend',
                               'toward', 'larger,', 'more', 'complex', 'brains', 'and', 'societies.\nEarly',
                               'hominins—particularly', 'the', 'australopithecines,', 'whose', 'brains', 'and',
                               'anatomy', 'are', 'in', 'many', 'ways', 'more', 'similar', 'to', 'ancestral',
                               'non-human', 'apes—are', 'less', 'often', 'referred', 'to', 'as', '"human"', 'than',
                               'hominins', 'of', 'the', 'genus', 'Homo.', 'Several', 'of', 'these', 'hominins', 'used',
                               'fire,', 'occupied', 'much', 'of', 'Eurasia,', 'and', 'gave', 'rise', 'to',
                               'anatomically', 'modern', 'Homo', 'sapiens', 'in', 'Africa', 'about', '200,000', 'years',
                               'ago.', 'They', 'began', 'to', 'exhibit', 'evidence', 'of', 'behavioral', 'modernity',
                               'around', '50,000', 'years', 'ago.', 'In', 'several', 'waves', 'of', 'migration,',
                               'anatomically', 'modern', 'humans', 'ventured', 'out', 'of', 'Africa', 'and',
                               'populated', 'most', 'of', 'the', 'world.\nThe', 'spread', 'of', 'humans', 'and',
                               'their', 'large', 'and', 'increasing', 'population', 'has', 'had', 'a', 'profound',
                               'impact', 'on', 'large', 'areas', 'of', 'the', 'environment', 'and', 'millions', 'of',
                               'native', 'species', 'worldwide.', 'Advantages', 'that', 'explain', 'this',
                               'evolutionary', 'success', 'include', 'a', 'relatively', 'larger', 'brain', 'with', 'a',
                               'particularly', 'well-developed', 'neocortex,', 'prefrontal', 'cortex', 'and',
                               'temporal', 'lobes,', 'which', 'enable', 'high', 'levels', 'of', 'abstract',
                               'reasoning,', 'language,', 'problem', 'solving,', 'sociality,', 'and', 'culture',
                               'through', 'social', 'learning.', 'Humans', 'use', 'tools', 'to', 'a', 'much', 'higher',
                               'degree', 'than', 'any', 'other', 'animal,', 'are', 'the', 'only', 'extant', 'species',
                               'known', 'to', 'build', 'fires', 'and', 'cook', 'their', 'food,', 'and', 'are', 'the',
                               'only', 'extant', 'species', 'to', 'clothe', 'themselves', 'and', 'create', 'and', 'use',
                               'numerous', 'other', 'technologies', 'and', 'arts.\nHumans', 'are', 'uniquely', 'adept',
                               'at', 'utilizing', 'systems', 'of', 'symbolic', 'communication', '(such', 'as',
                               'language', 'and', 'art)', 'for', 'self-expression', 'and', 'the', 'exchange', 'of',
                               'ideas,', 'and', 'for', 'organizing', 'themselves', 'into', 'purposeful', 'groups.',
                               'Humans', 'create', 'complex', 'social', 'structures', 'composed', 'of', 'many',
                               'cooperating', 'and', 'competing', 'groups,', 'from', 'families', 'and', 'kinship',
                               'networks', 'to', 'political', 'states.', 'Social', 'interactions', 'between', 'humans',
                               'have', 'established', 'an', 'extremely', 'wide', 'variety', 'of', 'values,', 'social',
                               'norms,', 'and', 'rituals,', 'which', 'together', 'form', 'the', 'basis', 'of', 'human',
                               'society.', 'Curiosity', 'and', 'the', 'human', 'desire', 'to', 'understand', 'and',
                               'influence', 'the', 'environment', 'and', 'to', 'explain', 'and', 'manipulate',
                               'phenomena', '(or', 'events)', 'has', 'provided', 'the', 'foundation', 'for',
                               'developing', 'science,', 'philosophy,', 'mythology,', 'religion,', 'anthropology,',
                               'and', 'numerous', 'other', 'fields', 'of', 'knowledge.\nThough', 'most', 'of', 'human',
                               'existence', 'has', 'been', 'sustained', 'by', 'hunting', 'and', 'gathering', 'in',
                               'band', 'societies,', 'increasing', 'numbers', 'of', 'human', 'societies', 'began', 'to',
                               'practice', 'sedentary', 'agriculture', 'approximately', 'some', '10,000', 'years',
                               'ago,', 'domesticating', 'plants', 'and', 'animals,', 'thus', 'allowing', 'for', 'the',
                               'growth', 'of', 'civilization.', 'These', 'human', 'societies', 'subsequently',
                               'expanded', 'in', 'size,', 'establishing', 'various', 'forms', 'of', 'government,',
                               'religion,', 'and', 'culture', 'around', 'the', 'world,', 'unifying', 'people', 'within',
                               'regions', 'to', 'form', 'states', 'and', 'empires.', 'The', 'rapid', 'advancement',
                               'of', 'scientific', 'and', 'medical', 'understanding', 'in', 'the', '19th', 'and',
                               '20th', 'centuries', 'led', 'to', 'the', 'development', 'of', 'fuel-driven',
                               'technologies', 'and', 'increased', 'lifespans,', 'causing', 'the', 'human',
                               'population', 'to', 'rise', 'exponentially.', 'Today', 'the', 'global', 'human',
                               'population', 'is', 'estimated', 'by', 'the', 'United', 'Nations', 'to', 'be', 'near',
                               '7.5', 'billion.']

    def test_find_nearest_neighbor_in_a_list(self):
        self.word2vec.pre_trained_google_news_300_model()
        self.neighbor = Neighborhood(self.word2vec.model, self.parameters)

        unigram = "human"

        min_distance = 0.3
        neighbor_size = 10

        neighbor = self.neighbor.find_nearest_neighbor_in_a_list(unigram, self.other_unigrams, min_distance,
                                                                 neighbor_size)

        self.assertEqual(neighbor, ['humans', 'sapiens', 'bipedal', 'anatomy', 'evolutionary', 'Humans', 'scientific'])

    def test_find_significant_neighbors(self):
        self.word2vec.pre_trained_google_news_300_model()
        self.neighbor = Neighborhood(self.word2vec.model, self.parameters)

        min_distance = 0.4
        neighbor_size = 5

        significant_neighbor = self.neighbor.find_significant_neighbors(self.other_unigrams, min_distance,
                                                                        neighbor_size)

        self.assertEqual(significant_neighbor,
                         [['Homo', 'sapiens', 'hominins', 'species', 'evolutionary'],
                          ['in', 'about', 'out', 'through', 'at'],
                          ['humans', 'sapiens', 'hominins', 'genus', 'evolutionary'],
                          ['the', 'only', 'other', 'that', 'this'],
                          ['humans', 'sapiens', 'human', 'bipedal', 'evolutionary'],
                          ['increased', 'trend', 'increasing', 'higher', 'growth'],
                          ['compared', 'rise', 'increasing', 'higher', 'expanded'],
                          ['more', 'less', 'than', 'large', 'higher'],
                          ['brains', 'anatomically', 'brain', 'prefrontal', 'temporal'],
                          ['are', 'has', 'this', 'been', 'be'],
                          ['had', 'have', 'been', 'subsequently', 'is']])

    def test_merge_close_neighbors(self):
        self.neighbor = Neighborhood(None, self.parameters)

        minimum_merge_intersection = 1
        merged_neighbors = self.neighbor.merge_close_neighbors(
            [['Homo', 'sapiens', 'hominins', 'species', 'evolutionary'],
             ['in', 'about', 'out', 'through', 'at'],
             ['humans', 'sapiens', 'hominins', 'genus', 'evolutionary'],
             ['the', 'only', 'other', 'that', 'this'],
             ['humans', 'sapiens', 'human', 'bipedal', 'evolutionary'],
             ['increased', 'trend', 'increasing', 'higher', 'growth'],
             ['compared', 'rise', 'increasing', 'higher', 'expanded'],
             ['more', 'less', 'than', 'large', 'higher'],
             ['brains', 'anatomically', 'brain', 'prefrontal', 'temporal'],
             ['are', 'has', 'this', 'been', 'be'],
             ['had', 'have', 'been', 'subsequently', 'is']], minimum_merge_intersection)

        self.assertEqual(merged_neighbors,
                         [{'species', 'hominins', 'Homo', 'bipedal', 'evolutionary', 'humans', 'genus', 'human',
                           'sapiens'}, {'out', 'through', 'in', 'at', 'about'},
                          {'is', 'are', 'been', 'other', 'this', 'be', 'that', 'had', 'the', 'only', 'have',
                           'subsequently', 'has'},
                          {'trend', 'higher', 'than', 'increasing', 'less', 'increased', 'compared', 'expanded', 'rise',
                           'more', 'growth', 'large'}, {'brain', 'brains', 'prefrontal', 'anatomically', 'temporal'}])

    def test_find_significant_merged_neighbors(self):
        self.word2vec.pre_trained_google_news_300_model()
        self.neighbor = Neighborhood(self.word2vec.model, self.parameters)

        min_distance = 0.4
        neighbor_size = 5
        minimum_merge_intersection = 1

        significant_merge_neighbor = self.neighbor.find_significant_merged_neighbors(self.other_unigrams, min_distance,
                                                                                     neighbor_size,
                                                                                     minimum_merge_intersection)

        expected_res = [{'sapiens', 'human', 'genus', 'evolutionary', 'species', 'humans', 'hominins', 'Homo',
                         'bipedal'}, {'this', 'other', 'be', 'are', 'that', 'only', 'been', 'has', 'the'},
                        {'less', 'expanded', 'than', 'rise', 'more', 'growth', 'higher', 'large', 'trend', 'compared',
                         'increasing', 'increased'}, {'subsequently', 'have', 'had', 'been', 'is'},
                        {'about', 'at', 'in', 'through', 'out'},
                        {'prefrontal', 'brain', 'brains', 'temporal', 'anatomically'}]

        self.assertTrue(significant_merge_neighbor[0] in expected_res)
        self.assertEqual(len(significant_merge_neighbor), len(expected_res))

    def test_remove_stopwords_neighbors(self):
        self.neighbor = Neighborhood(None, self.parameters)
        max_stop_words = 3
        neighbors = [{'sapiens', 'human', 'genus', 'evolutionary', 'species', 'humans', 'hominins', 'Homo',
                      'bipedal'}, {'this', 'other', 'be', 'are', 'that', 'only', 'been', 'has', 'the'},
                     {'less', 'expanded', 'than', 'rise', 'more', 'growth', 'higher', 'large', 'trend', 'compared',
                      'increasing', 'increased'}, {'subsequently', 'have', 'had', 'been', 'is'},
                     {'about', 'at', 'in', 'through', 'out'},
                     {'prefrontal', 'brain', 'brains', 'temporal', 'anatomically'}]
        res = self.neighbor.remove_stopwords_neighbors(neighbors, max_stop_words)
        expected_res = [
            {'hominins', 'Homo', 'sapiens', 'evolutionary', 'genus', 'humans', 'human', 'bipedal', 'species'},
            {'increasing', 'less', 'trend', 'increased', 'higher', 'compared', 'rise', 'expanded', 'large', 'growth'},
            {'anatomically', 'brain', 'prefrontal', 'brains', 'temporal'}]
        self.assertEqual(res, expected_res)

    def test_remove_stemmed_similar_words(self):
        self.neighbor = Neighborhood(None, self.parameters)
        neighbors = [{'sapiens', 'human', 'genus', 'evolutionary', 'species', 'humans', 'hominins', 'Homo',
                      'bipedal'}, {'this', 'other', 'be', 'are', 'that', 'only', 'been', 'has', 'the'},
                     {'less', 'expanded', 'than', 'rise', 'more', 'growth', 'higher', 'large', 'trend', 'compared',
                      'increasing', 'increased'}, {'subsequently', 'have', 'had', 'been', 'is'},
                     {'about', 'at', 'in', 'through', 'out'},
                     {'prefrontal', 'brain', 'brains', 'temporal', 'anatomically'}]
        res = self.neighbor.remove_stemmed_similar_words(neighbors)
        self.assertEqual(res, [{'bipedal', 'species', 'human', 'sapiens', 'hominins', 'Homo', 'genus', 'evolutionary'},
                               {'been', 'the', 'are', 'that', 'other', 'be', 'has', 'this', 'only'},
                               {'increasing', 'than', 'rise', 'trend', 'growth', 'higher', 'more', 'compared', 'less',
                                'expanded', 'large'}, {'been', 'is', 'subsequently', 'have', 'had'},
                               {'at', 'about', 'through', 'in', 'out'},
                               {'brains', 'temporal', 'anatomically', 'prefrontal'}])
