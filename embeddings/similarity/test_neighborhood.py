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

        self.significant_neighbors_wiki = [{'revealed', 'arose', 'set', 'arrived', 'caught', 'began', 'came', 'moved',
                                            'become'},
                                           {'major', 'top', 'considered', 'certain', 'expects', 'decided', 'influenced',
                                            'depending',
                                            'identified', 'unable', 'driven', 'hopes', 'failed', 'dictated', 'reputed',
                                            'intention', 'led',
                                            'unlikely', 'renowned', 'aided', 'believe', 'counteracted', 'regardless',
                                            'key',
                                            'trying',
                                            'described', 'tried', 'promised', 'managed', 'inspired', 'either', 'know',
                                            'determined'},
                                           {'experience', 'background', 'experienced', 'seasoned', 'accomplishments',
                                            'vocabulary',
                                            'abilities', 'qualities', 'humility', 'characteristic', 'knowledge',
                                            'trait'},
                                           {'greatly', 'dramatic', 'major', 'rapidly', 'classic', 'slightly',
                                            'memorable',
                                            'considerably',
                                            'key', 'amusing', 'notable', 'beautiful', 'significant'},
                                           {'companionate', 'designs', 'confusion', 'environment', 'pliancy',
                                            'adaptations',
                                            'literature',
                                            'contemporary', 'languages', 'society', 'shades', 'prejudice', 'wrongs',
                                            'text',
                                            'themes',
                                            'distrust', 'gentility', 'perception', 'playfulness', 'manuscript',
                                            'fashion'},
                                           {'darcy', 'enact', 'gentleman', 'lizzy', 'adopts', 'approve', 'lord', 'ask',
                                            'seek'},
                                           {'introduced', 'acquired', 'expanded', 'formerly', 'name'},
                                           {'matters', 'mistake', 'concern', 'obstacle', 'factors', 'situation',
                                            'problem',
                                            'themes',
                                            'issues'},
                                           {'saw', 'acknowledged', 'really', 'knew', 'gossiping', 'silly', 'sarcastic',
                                            'cried',
                                            'jokes'},
                                           {'avoid', 'persuade', 'ease', 'tempt', 'improve', 'still', 'put',
                                            'encourage', 'stays',
                                            'raise',
                                            'dissuade', 'keep', 'entice'},
                                           {'approach', 'process', 'formula', 'concept', 'technique', 'viewpoint',
                                            'way',
                                            'method',
                                            'tool'},
                                           {'receive', 'undertaken', 'done', 'serves', 'granted', 'uses', 'gives',
                                            'learns',
                                            'takes',
                                            'creates', 'given', 'confronts', 'exposes', 'carried', 'providing',
                                            'conducted',
                                            'undergone',
                                            'allowed', 'taken', 'brought', 'responds'},
                                           {'deeply', 'introspective', 'feelings', 'economically', 'emotional'},
                                           {'one', 'three', 'minimum', 'six', 'five'},
                                           {'journal', 'released', 'novelist', 'reads', 'mentioned', 'wrote', 'spoke',
                                            'sketching',
                                            'speaking', 'spoken', 'article', 'talk', 'essay', 'amorous', 'witty',
                                            'written',
                                            'artless',
                                            'writing', 'published', 'literary'},
                                           {'thoroughly', 'closely', 'beautifully', 'heavily', 'scrupulously'},
                                           {'economy', 'company', 'commercial', 'financial', 'market'},
                                           {'wealth', 'rich', 'wealthy', 'prosperous', 'income'},
                                           {'initially', 'later', 'previously', 'recently', 'subsequently'},
                                           {'ignorance', 'disappointment', 'upset', 'surprised', 'agonizing',
                                            'impropriety',
                                            'insolence',
                                            'decorum', 'decency', 'ridicule', 'indecorous', 'rectitude', 'horrified',
                                            'embarrassment',
                                            'disgrace', 'arrogance', 'folly'},
                                           {'adult', 'women', 'others', 'children', 'child', 'men', 'social',
                                            'upbringing',
                                            'anybody'},
                                           {'learns', 'goes', 'seems', 'comes', 'happened'},
                                           {'flirting', 'romances', 'connections', 'affair', 'jealous',
                                            'interrelationships',
                                            'friendship',
                                            'elope', 'mingling', 'seducing', 'interactions', 'amorous', 'laughing'},
                                           {'portrait', 'depiction', 'picture', 'illustration', 'graphic', 'example',
                                            'artwork',
                                            'gallery'},
                                           {'illogical', 'actually', 'truth', 'irony', 'silly', 'reality', 'haste',
                                            'fact',
                                            'cynical',
                                            'quick', 'great', 'condescending', 'childish', 'merely', 'unnecessary',
                                            'lies',
                                            'genuine',
                                            'serious', 'true', 'real'},
                                           {'unmarried', 'sexual', 'marriage', 'male', 'adultery'},
                                           {'journal', 'memoir', 'sequel', 'music', 'trilogy', 'article', 'concert',
                                            'manuscript',
                                            'writing'},
                                           {'admit', 'evidence', 'proof', 'case', 'believe', 'reasons', 'argument',
                                            'empirical',
                                            'justification', 'say', 'attest', 'motive'},
                                           {'forced', 'due', 'caused', 'fact', 'nevertheless', 'resulted', 'spite',
                                            'despite'},
                                           {'feminity', 'amiability', 'gentility', 'playfulness', 'insolence'},
                                           {'particularly', 'sufficiently', 'extremely', 'tendency', 'often'},
                                           {'involvement', 'importance', 'part', 'position', 'instrumental'},
                                           {'music', 'piano', 'imitating', 'dance', 'harp'},
                                           {'biology', 'scientists', 'scholarly', 'research', 'empirical'},
                                           {'back', 'apart', 'around', 'behind', 'aside'},
                                           {'barely', 'every', 'hardly', 'though', 'practically'},
                                           {'hurting', 'sickly', 'low', 'bad', 'poor'},
                                           {'imperative', 'stresses', 'advantages', 'role', 'necessity'},
                                           {'unequal', 'overbearing', 'illiberal', 'subjection', 'evil'},
                                           {'respect', 'affection', 'confidence', 'veneration', 'regard'}]

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
        res = self.neighbor.remove_stemmed_similar_words_neighbors(neighbors)
        expected_res = [{'bipedal', 'species', 'human', 'sapiens', 'hominins', 'Homo', 'genus', 'evolutionary'},
                        {'been', 'the', 'are', 'that', 'other', 'be', 'has', 'this', 'only'},
                        {'increasing', 'than', 'rise', 'trend', 'growth', 'higher', 'more', 'compared', 'less',
                         'expanded', 'large'}, {'been', 'is', 'subsequently', 'have', 'had'},
                        {'at', 'about', 'through', 'in', 'out'},
                        {'brains', 'temporal', 'anatomically', 'prefrontal'}]
        self.assertEqual(res[0], expected_res[0])
        self.assertEqual(len(res), len(expected_res))

    def test_remove_stemmed_similar_words_list(self):
        self.neighbor = Neighborhood(None, self.parameters)
        neighbor = ['sapiens', 'human', 'genus', 'evolutionary', 'species', 'humans', 'hominins', 'Homo', 'bipedal']
        res = self.neighbor.remove_stemmed_similar_words_list(neighbor)
        expected_res = ['bipedal', 'species', 'human', 'sapiens', 'hominins', 'Homo', 'genus', 'evolutionary']
        self.assertEqual(set(res), set(expected_res))

    def test_replace_stemmed_similar_words_list(self):
        self.neighbor = Neighborhood(None, self.parameters)
        neighbor = ['sapiens', 'human', 'genus', 'evolutionary', 'species', 'humans', "Human", 'hominins', 'Homo',
                    'bipedal']
        res = self.neighbor.replace_stemmed_similar_words_list(neighbor)
        expected_res = ['sapiens', 'human', 'genus', 'evolutionary', 'species', 'human', 'human', 'hominins', 'Homo',
                        'bipedal']
        self.assertEqual(res, expected_res)

    def test_find_significant_pruned_neighbors(self):
        self.word2vec.pre_trained_google_news_300_model()
        self.neighbor = Neighborhood(self.word2vec.model, self.parameters)

        min_distance = 0.4
        neighbor_size = 5
        minimum_merge_intersection = 1
        max_stop_words = 1

        res = self.neighbor.find_significant_pruned_neighbors(self.other_unigrams, min_distance,
                                                              neighbor_size,
                                                              minimum_merge_intersection,
                                                              max_stop_words)

        expected_res = [{'humans', 'sapiens', 'hominins', 'species', 'evolutionary', 'genus', 'Homo', 'bipedal'},
                        {'brains', 'anatomically', 'prefrontal', 'temporal'}]

        self.assertTrue(len(res[0]) == len(expected_res[0]) or len(res[0]) == len(expected_res[1]))
        self.assertTrue(len(res[1]) == len(expected_res[1]) or len(res[1]) == len(expected_res[0]))

    def test_find_significant_pruned_neighbors_in_doc(self):
        self.word2vec.pre_trained_google_news_300_model()
        self.neighbor = Neighborhood(self.word2vec.model, self.parameters)

        min_distance = 0.4
        neighbor_size = 5
        minimum_merge_intersection = 1
        max_stop_words = 1

        res = self.neighbor.find_significant_pruned_neighbors_in_doc(
            "../../configs/others/pride_and_prejudice_wiki.txt",
            min_distance,
            neighbor_size,
            minimum_merge_intersection,
            max_stop_words)

        expected_res = self.significant_neighbors_wiki

        self.assertEqual(len(res), len(expected_res))

    def test_find_significant_neighbors_weight(self):
        self.parameters.params["repo_dir"] = '/scratch/index/indri_5_7/ap8889'
        self.neighbor = Neighborhood(None, self.parameters)
        doc_words = self.neighbor.get_words("../../configs/others/pride_and_prejudice_wiki.txt")
        significant_neighbors_wiki = self.neighbor.index_neighbors(self.significant_neighbors_wiki)
        significant_neighbors_weight = self.neighbor.find_significant_neighbors_weight(doc_words,
                                                                                       significant_neighbors_wiki)
        print(significant_neighbors_weight, file=sys.stderr)
        self.assertEqual(significant_neighbors_weight,
                         {0: 1.9657078410960527, 1: 1.9683658951704364, 2: 2.4480979287834641, 3: 2.0842528829702265,
                          4: 2.7511415404138724, 5: 2.7896288147053578, 6: 2.4429873399337798, 7: 1.5525003405814493,
                          8: 3.3854693632901158, 9: 2.2115369760229528, 10: 2.0384620803628803, 11: 1.4874710681311145,
                          12: 2.1526264204636982, 13: 2.0828900357613529, 14: 2.706715019070205, 15: 2.6264039661400878,
                          16: 1.203510833885312, 17: 2.2288942504357725, 18: 1.4284576106213134, 19: 3.2435934976361729,
                          20: 2.7711783428977199, 21: 2.2893316431048722, 22: 3.1796463326772111,
                          23: 2.4549643568113124, 24: 2.5434144735503503, 25: 2.7581736015111566, 26: 2.564925624408728,
                          27: 1.8296276817113244, 28: 2.0051271018302055, 29: 4.006608961860346, 30: 2.9067522415376059,
                          31: 1.278885369636712, 32: 2.7560488141020385, 33: 2.4642912491882978, 34: 4.1549814591354233,
                          35: 4.3677319378719091, 36: 2.0299240134899401, 37: 2.2725138641549796, 38: 3.597216546805833,
                          39: 1.9776010225667644})

    def test_sort_significant_neighbors(self):
        self.neighbor = Neighborhood(None, self.parameters)

        significant_neighbors_wiki = {
            0: {'caught', 'set', 'came', 'revealed', 'moved', 'began', 'arose', 'become', 'arrived'},
            1: {'failed', 'expects', 'described', 'major', 'identified', 'counteracted', 'hopes', 'believe', 'driven',
                'reputed', 'considered', 'dictated', 'led', 'determined', 'either', 'managed', 'aided', 'inspired',
                'know',
                'decided', 'renowned', 'promised', 'intention', 'unlikely', 'top', 'influenced', 'unable', 'depending',
                'regardless', 'tried', 'certain', 'trying', 'key'},
            2: {'trait', 'humility', 'characteristic', 'qualities', 'knowledge', 'accomplishments', 'vocabulary',
                'seasoned', 'experience', 'experienced', 'background', 'abilities'},
            3: {'classic', 'significant', 'memorable', 'beautiful', 'major', 'rapidly', 'amusing', 'notable',
                'slightly',
                'key', 'greatly', 'dramatic', 'considerably'},
            4: {'contemporary', 'prejudice', 'gentility', 'wrongs', 'companionate', 'text', 'themes', 'distrust',
                'fashion', 'manuscript', 'literature', 'adaptations', 'perception', 'playfulness', 'environment',
                'shades',
                'confusion', 'pliancy', 'languages', 'designs', 'society'},
            5: {'approve', 'darcy', 'seek', 'lizzy', 'ask', 'gentleman', 'adopts', 'enact', 'lord'},
            6: {'introduced', 'acquired', 'formerly', 'expanded', 'name'},
            7: {'factors', 'concern', 'mistake', 'problem', 'matters', 'situation', 'issues', 'obstacle', 'themes'},
            8: {'jokes', 'silly', 'sarcastic', 'saw', 'knew', 'gossiping', 'acknowledged', 'really', 'cried'},
            9: {'persuade', 'raise', 'ease', 'avoid', 'entice', 'put', 'dissuade', 'keep', 'encourage', 'improve',
                'tempt',
                'stays', 'still'},
            10: {'way', 'viewpoint', 'process', 'technique', 'tool', 'approach', 'concept', 'method', 'formula'},
            11: {'granted', 'brought', 'exposes', 'done', 'gives', 'undergone', 'uses', 'given', 'serves', 'undertaken',
                 'confronts', 'creates', 'providing', 'takes', 'allowed', 'carried', 'responds', 'conducted', 'learns',
                 'receive', 'taken'}, 12: {'feelings', 'economically', 'introspective', 'emotional', 'deeply'},
            13: {'one', 'six', 'minimum', 'five', 'three'},
            14: {'mentioned', 'spoke', 'written', 'wrote', 'essay', 'witty', 'writing', 'literary', 'amorous',
                 'released',
                 'reads', 'speaking', 'journal', 'sketching', 'published', 'article', 'talk', 'artless', 'spoken',
                 'novelist'}, 15: {'thoroughly', 'closely', 'heavily', 'beautifully', 'scrupulously'},
            16: {'company', 'financial', 'commercial', 'economy', 'market'},
            17: {'wealthy', 'rich', 'wealth', 'income', 'prosperous'},
            18: {'subsequently', 'initially', 'recently', 'later', 'previously'},
            19: {'ignorance', 'agonizing', 'disappointment', 'impropriety', 'decorum', 'decency', 'embarrassment',
                 'disgrace', 'arrogance', 'upset', 'horrified', 'surprised', 'folly', 'insolence', 'indecorous',
                 'rectitude', 'ridicule'},
            20: {'women', 'children', 'social', 'others', 'men', 'upbringing', 'anybody', 'child', 'adult'},
            21: {'goes', 'happened', 'seems', 'learns', 'comes'},
            22: {'romances', 'elope', 'friendship', 'jealous', 'affair', 'mingling', 'interactions', 'amorous',
                 'laughing',
                 'flirting', 'seducing', 'interrelationships', 'connections'},
            23: {'depiction', 'picture', 'artwork', 'graphic', 'illustration', 'portrait', 'gallery', 'example'},
            24: {'silly', 'genuine', 'lies', 'condescending', 'fact', 'childish', 'unnecessary', 'true', 'cynical',
                 'illogical', 'haste', 'truth', 'serious', 'reality', 'merely', 'quick', 'irony', 'real', 'actually',
                 'great'}, 25: {'sexual', 'male', 'marriage', 'unmarried', 'adultery'},
            26: {'manuscript', 'trilogy', 'writing', 'memoir', 'music', 'journal', 'concert', 'article', 'sequel'},
            27: {'empirical', 'case', 'evidence', 'admit', 'justification', 'attest', 'motive', 'reasons', 'say',
                 'argument', 'believe', 'proof'},
            28: {'spite', 'despite', 'nevertheless', 'fact', 'due', 'caused', 'resulted', 'forced'},
            29: {'gentility', 'amiability', 'playfulness', 'feminity', 'insolence'},
            30: {'tendency', 'sufficiently', 'often', 'extremely', 'particularly'},
            31: {'importance', 'instrumental', 'involvement', 'position', 'part'},
            32: {'dance', 'imitating', 'piano', 'music', 'harp'},
            33: {'empirical', 'scholarly', 'scientists', 'biology', 'research'},
            34: {'aside', 'apart', 'back', 'around', 'behind'},
            35: {'barely', 'practically', 'though', 'every', 'hardly'},
            36: {'poor', 'sickly', 'low', 'bad', 'hurting'},
            37: {'imperative', 'stresses', 'necessity', 'role', 'advantages'},
            38: {'unequal', 'subjection', 'illiberal', 'evil', 'overbearing'},
            39: {'regard', 'respect', 'veneration', 'confidence', 'affection'}}
        significant_neighbors_weight = {0: 1.9657078410960527, 1: 1.9683658951704364, 2: 2.4480979287834641,
                                        3: 2.0842528829702265, 4: 2.7511415404138724, 5: 2.7896288147053578,
                                        6: 2.4429873399337798, 7: 1.5525003405814493, 8: 3.3854693632901158,
                                        9: 2.2115369760229528, 10: 2.0384620803628803, 11: 1.4874710681311145,
                                        12: 2.1526264204636982, 13: 2.0828900357613529, 14: 2.706715019070205,
                                        15: 2.6264039661400878, 16: 1.203510833885312, 17: 2.2288942504357725,
                                        18: 1.4284576106213134, 19: 3.2435934976361729, 20: 2.7711783428977199,
                                        21: 2.2893316431048722, 22: 3.1796463326772111, 23: 2.4549643568113124,
                                        24: 2.5434144735503503, 25: 2.7581736015111566, 26: 2.564925624408728,
                                        27: 1.8296276817113244, 28: 2.0051271018302055, 29: 4.006608961860346,
                                        30: 2.9067522415376059, 31: 1.278885369636712, 32: 2.7560488141020385,
                                        33: 2.4642912491882978, 34: 4.1549814591354233, 35: 4.3677319378719091,
                                        36: 2.0299240134899401, 37: 2.2725138641549796, 38: 3.597216546805833,
                                        39: 1.9776010225667644}
        sorted_significant_neighbors = self.neighbor.sort_significant_neighbors(significant_neighbors_weight,
                                                                                significant_neighbors_wiki)
        print(sorted_significant_neighbors, file=sys.stderr)
        self.assertEqual(sorted_significant_neighbors,
                         [({'practically', 'every', 'barely', 'hardly', 'though'}, 4.367731937871909),
                          ({'around', 'aside', 'apart', 'behind', 'back'}, 4.154981459135423),
                          ({'gentility', 'playfulness', 'amiability', 'insolence', 'feminity'}, 4.006608961860346),
                          ({'unequal', 'subjection', 'evil', 'overbearing', 'illiberal'}, 3.597216546805833), (
                          {'sarcastic', 'silly', 'acknowledged', 'really', 'saw', 'cried', 'jokes', 'knew',
                           'gossiping'}, 3.385469363290116), (
                          {'folly', 'impropriety', 'ridicule', 'surprised', 'horrified', 'embarrassment',
                           'disappointment', 'insolence', 'decency', 'upset', 'arrogance', 'agonizing', 'ignorance',
                           'disgrace', 'decorum', 'rectitude', 'indecorous'}, 3.243593497636173), (
                          {'amorous', 'mingling', 'laughing', 'romances', 'interrelationships', 'flirting', 'jealous',
                           'friendship', 'affair', 'seducing', 'interactions', 'connections', 'elope'},
                          3.179646332677211),
                          ({'often', 'tendency', 'particularly', 'sufficiently', 'extremely'}, 2.906752241537606), (
                          {'approve', 'lizzy', 'ask', 'darcy', 'seek', 'gentleman', 'adopts', 'enact', 'lord'},
                          2.789628814705358), (
                          {'social', 'upbringing', 'women', 'anybody', 'child', 'children', 'men', 'adult', 'others'},
                          2.77117834289772),
                          ({'marriage', 'unmarried', 'male', 'sexual', 'adultery'}, 2.7581736015111566),
                          ({'music', 'piano', 'harp', 'imitating', 'dance'}, 2.7560488141020385), (
                          {'gentility', 'perception', 'contemporary', 'companionate', 'literature', 'environment',
                           'distrust', 'text', 'playfulness', 'themes', 'prejudice', 'manuscript', 'fashion',
                           'confusion', 'languages', 'shades', 'designs', 'society', 'adaptations', 'wrongs',
                           'pliancy'}, 2.7511415404138724), (
                          {'amorous', 'literary', 'reads', 'article', 'witty', 'artless', 'mentioned', 'written',
                           'sketching', 'talk', 'spoken', 'novelist', 'speaking', 'writing', 'published', 'journal',
                           'spoke', 'wrote', 'released', 'essay'}, 2.706715019070205),
                          ({'closely', 'heavily', 'thoroughly', 'scrupulously', 'beautifully'}, 2.6264039661400878), (
                          {'sequel', 'manuscript', 'article', 'music', 'memoir', 'writing', 'concert', 'journal',
                           'trilogy'}, 2.564925624408728), (
                          {'true', 'great', 'childish', 'unnecessary', 'silly', 'haste', 'real', 'fact', 'illogical',
                           'genuine', 'merely', 'condescending', 'reality', 'quick', 'truth', 'irony', 'cynical',
                           'lies', 'actually', 'serious'}, 2.5434144735503503),
                          ({'research', 'empirical', 'biology', 'scholarly', 'scientists'}, 2.464291249188298), (
                          {'picture', 'illustration', 'example', 'graphic', 'portrait', 'gallery', 'artwork',
                           'depiction'}, 2.4549643568113124), (
                          {'experienced', 'characteristic', 'vocabulary', 'qualities', 'accomplishments', 'knowledge',
                           'abilities', 'trait', 'background', 'seasoned', 'experience', 'humility'},
                          2.448097928783464),
                          ({'name', 'introduced', 'expanded', 'formerly', 'acquired'}, 2.44298733993378),
                          ({'happened', 'seems', 'learns', 'comes', 'goes'}, 2.2893316431048722),
                          ({'advantages', 'role', 'stresses', 'necessity', 'imperative'}, 2.2725138641549796),
                          ({'wealth', 'rich', 'wealthy', 'income', 'prosperous'}, 2.2288942504357725), (
                          {'dissuade', 'encourage', 'ease', 'raise', 'tempt', 'persuade', 'put', 'stays', 'entice',
                           'improve', 'avoid', 'still', 'keep'}, 2.211536976022953),
                          ({'introspective', 'emotional', 'feelings', 'deeply', 'economically'}, 2.1526264204636982), (
                          {'classic', 'memorable', 'notable', 'slightly', 'amusing', 'beautiful', 'significant',
                           'greatly', 'considerably', 'key', 'rapidly', 'dramatic', 'major'}, 2.0842528829702265),
                          ({'one', 'three', 'six', 'five', 'minimum'}, 2.082890035761353), (
                          {'viewpoint', 'approach', 'formula', 'way', 'technique', 'concept', 'tool', 'process',
                           'method'}, 2.0384620803628803),
                          ({'bad', 'low', 'sickly', 'poor', 'hurting'}, 2.02992401348994), (
                          {'caused', 'resulted', 'nevertheless', 'fact', 'despite', 'forced', 'due', 'spite'},
                          2.0051271018302055),
                          ({'respect', 'confidence', 'regard', 'affection', 'veneration'}, 1.9776010225667644), (
                          {'identified', 'regardless', 'counteracted', 'renowned', 'depending', 'expects', 'considered',
                           'driven', 'hopes', 'led', 'unable', 'major', 'unlikely', 'tried', 'believe', 'intention',
                           'dictated', 'inspired', 'know', 'trying', 'promised', 'described', 'top', 'determined',
                           'certain', 'managed', 'reputed', 'key', 'decided', 'aided', 'failed', 'either',
                           'influenced'}, 1.9683658951704364), (
                          {'arrived', 'came', 'become', 'arose', 'caught', 'set', 'revealed', 'moved', 'began'},
                          1.9657078410960527), (
                          {'reasons', 'motive', 'attest', 'proof', 'case', 'believe', 'admit', 'empirical', 'evidence',
                           'justification', 'say', 'argument'}, 1.8296276817113244), (
                          {'matters', 'issues', 'factors', 'situation', 'themes', 'concern', 'problem', 'mistake',
                           'obstacle'}, 1.5525003405814493), (
                          {'brought', 'confronts', 'allowed', 'receive', 'taken', 'learns', 'uses', 'exposes',
                           'undertaken', 'serves', 'gives', 'done', 'granted', 'given', 'carried', 'responds', 'takes',
                           'conducted', 'undergone', 'providing', 'creates'}, 1.4874710681311145),
                          ({'later', 'subsequently', 'previously', 'recently', 'initially'}, 1.4284576106213134),
                          ({'position', 'part', 'instrumental', 'importance', 'involvement'}, 1.278885369636712),
                          ({'financial', 'commercial', 'market', 'economy', 'company'}, 1.203510833885312)])
