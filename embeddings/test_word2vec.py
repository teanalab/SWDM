import sys
from unittest import TestCase

from embeddings.word2vec import Word2vec


class TestWord2vec(TestCase):
    def setUp(self):
        self.word2vec = Word2vec()

    def test_gen_similar_words(self):
        self.word2vec.pre_trained_google_news_300_model()

        res = self.word2vec.gen_similar_words('human', 20)

        print(res, file=sys.stderr)

        expected_res = [(u'human_beings', 0.613968014717102), (u'humans', 0.5917960405349731),
                        (u'impertinent_flamboyant_endearingly', 0.5868302583694458),
                        (u'employee_Laura_Althouse', 0.5639358758926392),
                        (u'humankind', 0.5636305809020996), (u'Human', 0.5524993538856506),
                        (u'mankind', 0.5346406698226929), (u'Christine_Gaugler_head', 0.5272535681724548),
                        (u'humanity', 0.5262271165847778), (u'sentient_intelligent', 0.5201493501663208),
                        (u'nonhuman', 0.5158316493034363), (u'nonhuman_animals', 0.5148903131484985),
                        (u'growth_hormone_misbranding', 0.5118885040283203), (u'nonhuman_species', 0.5054227113723755),
                        (u'tiniest_fragments', 0.5019494891166687), (u'animal', 0.4987776577472687),
                        (u'Treating_Gina', 0.4941716492176056), (u'Broadcastr_seeks', 0.4918348491191864),
                        (u'Stephen_Mimnaugh', 0.4915102422237396), (u'sentient_beings', 0.4901559352874756)]

        self.assertEqual(res, expected_res)

        res = self.word2vec.gen_similar_words('Airbus', 20)

        print(res, file=sys.stderr)
        expected_res = [('Boeing', 0.761221170425415), ('Airbus_SAS', 0.7509450912475586),
                        ('planemaker', 0.7396568059921265), ('AirbusAirbus', 0.7212362885475159),
                        ('Boeing_BA.N', 0.7118062376976013), ('Airbus_A###', 0.7087818384170532),
                        ('EADS', 0.6971980333328247), ('Airbus_EAD.PA', 0.6961885690689087),
                        ('Airbus_Industrie', 0.6937824487686157), ('European_planemaker', 0.6925910711288452),
                        ('European_planemaker_Airbus', 0.6860209703445435), ('Dreamliner', 0.6797409057617188),
                        ('Airbus_A###_superjumbo', 0.6710634827613831), ('EADS_Airbus', 0.6707702875137329),
                        ('CSeries', 0.6678756475448608), ('Embraer', 0.6677765846252441),
                        ('Dassault_Aviation', 0.667271077632904), ('airframer', 0.6671715974807739),
                        ('A###_XWB', 0.6649736762046814), ('Airbus_jets', 0.6602391004562378)]

        self.assertEqual(res, expected_res)

        res = self.word2vec.gen_similar_words('and', 20)
        self.assertEqual(res, [])
