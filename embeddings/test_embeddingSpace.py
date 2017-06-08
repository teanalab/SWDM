import time
from unittest import TestCase

from embeddings.embedding_space import EmbeddingSpace
from parameters.parameters import Parameters

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class TestEmbeddingSpace(TestCase):
    def setUp(self):
        self.parameters = Parameters()
        self.parameters.params["word2vec"] = {"upper_threshold": 0.8, "lower_threshold": 0.6, "n_max": 5}
        self.parameters.params["repo_dir"] = '../index/test_files/index'

    def test_find_unigrams_in_embedding_space(self):
        embedding_space = EmbeddingSpace(self.parameters)
        embedding_space.initialize()
        unigrams = embedding_space.find_unigrams_in_embedding_space("hello world how are you")
        unigrams_expected = [[('hello', 1), ('hi', 0.6548984050750732), ('goodbye', 0.639905571937561),
                              ('howdy', 0.6310956478118896)],
                             [('world', 1), ('globe', 0.6945997476577759), ('theworld', 0.6902236938476562)],
                             [('how', 1), ('what', 0.6820360422134399)], [('are', 1), ('were', 0.7415369153022766)],
                             [('you', 1), ('your', 0.7808908224105835), ('yourself', 0.7698668241500854),
                              ('I', 0.6739810109138489), ('we', 0.6565826535224915), ('somebody', 0.6341674327850342)]]
        self.assertEqual(unigrams, unigrams_expected)

        unigrams = embedding_space.find_unigrams_in_embedding_space("Airbus")
        unigrams_expected = [[('hello', 1), ('hi', 0.6548984050750732), ('goodbye', 0.639905571937561),
                              ('howdy', 0.6310956478118896)],
                             [('world', 1), ('globe', 0.6945997476577759), ('theworld', 0.6902236938476562)],
                             [('how', 1), ('what', 0.6820360422134399)], [('are', 1), ('were', 0.7415369153022766)],
                             [('you', 1), ('your', 0.7808908224105835), ('yourself', 0.7698668241500854),
                              ('I', 0.6739810109138489), ('we', 0.6565826535224915), ('somebody', 0.6341674327850342)]]
        self.assertEqual(unigrams, unigrams_expected)

    def test_check_if_already_stem_exists(self):
        embedding_space = EmbeddingSpace(self.parameters)

        unigrams_in_embedding_space_pruned = [('you', 1), ('your', 0.7808908224105835),
                                              ('yourself', 0.7698668241500854), ('I', 0.6739810109138489),
                                              ('we', 0.6565826535224915), ('somebody', 0.6341674327850342)]
        unigram = "Where"
        orig_unigram = "You"
        res = embedding_space.check_if_already_stem_exists(unigrams_in_embedding_space_pruned, unigram, orig_unigram)
        self.assertEqual(res, False)

        unigram = "YOU"
        orig_unigram = "You"
        res = embedding_space.check_if_already_stem_exists(unigrams_in_embedding_space_pruned, unigram, orig_unigram)
        self.assertEqual(res, True)

    def test_gen_similar_words(self):
        embedding_space = EmbeddingSpace(self.parameters)
        embedding_space.initialize()

        t = time.process_time()
        res = embedding_space.gen_similar_words('human', embedding_space.word2vec)
        elapsed_time_1 = time.process_time() - t

        expected_res = [('human_beings', 0.613968014717102), ('humans', 0.5917960405349731),
                        ('impertinent_flamboyant_endearingly', 0.5868302583694458),
                        ('employee_Laura_Althouse', 0.5639358758926392), ('humankind', 0.5636305809020996),
                        ('Human', 0.5524993538856506), ('mankind', 0.5346406698226929),
                        ('Christine_Gaugler_head', 0.5272535681724548), ('humanity', 0.5262271165847778),
                        ('sentient_intelligent', 0.5201493501663208), ('nonhuman', 0.5158316493034363),
                        ('nonhuman_animals', 0.5148903131484985), ('growth_hormone_misbranding', 0.5118885040283203),
                        ('nonhuman_species', 0.5054227113723755), ('tiniest_fragments', 0.5019494891166687),
                        ('animal', 0.4987776577472687), ('Treating_Gina', 0.4941716492176056),
                        ('Broadcastr_seeks', 0.4918348491191864), ('Stephen_Mimnaugh', 0.4915102422237396),
                        ('sentient_beings', 0.4901559352874756), ('bodily', 0.48965349793434143),
                        ('nonhumans', 0.48545318841934204), ('Douglas_Interiano_spokesman', 0.4853293299674988),
                        ('intelligence_HUMINT', 0.48333364725112915), ('®_Nasdaq_KNXA', 0.4819592237472534),
                        ('machine_interfaces_HMIs', 0.48134148120880127),
                        ('evolutionary_adaptations', 0.47514891624450684), ('interfaces_HMI', 0.47141575813293457),
                        ('homo_sapiens', 0.46880772709846497), ('beings', 0.4686187505722046),
                        ('Human_beings', 0.46829017996788025), ('mankinds', 0.4668985605239868),
                        ('finiteness', 0.4645306468009949), ('executive_Nancy_Tullos', 0.4604729413986206),
                        ('inanimate', 0.45744433999061584), ('prehuman', 0.4573291540145874),
                        ('transmissable', 0.4564620852470398), ('embryo_clones', 0.45624130964279175),
                        ('fallenness', 0.4560883939266205), ('addition_Exponent_evaluates', 0.45543545484542847),
                        ('sentient_creature', 0.45398035645484924), ('sentient', 0.45397087931632996),
                        ('bullet_shell_casing', 0.4528302550315857), ('Humans', 0.4526465833187103),
                        ('Addex_Pharmaceuticals_www.addexpharma.com_discovers', 0.45002347230911255),
                        ('John_L._VandeBerg', 0.44979870319366455), ('natural', 0.44833338260650635),
                        ('displaying_depraved_indifference', 0.4483269453048706), ('humanness', 0.4478735327720642),
                        ('sentience', 0.44722604751586914), ('Savvion_BPM_platform', 0.4471215009689331),
                        ('superorganism', 0.4470677971839905), ('NASDAQ_KNXA_global', 0.4459344148635864),
                        ('vertebrate_animals', 0.445374071598053), ('nonliving', 0.44524049758911133),
                        ('spiritual_beings', 0.44508519768714905), ('comparative_anatomy', 0.44482192397117615),
                        ('creaturely', 0.44442033767700195), ('chimp_genomes', 0.4440094232559204),
                        ('Patricia_Prue', 0.4430835247039795), ('mortal_beings', 0.4427467882633209),
                        ('finitude', 0.44232380390167236), ('bipedal_locomotion', 0.4422861635684967),
                        ('totipotent', 0.44224387407302856), ('animality', 0.4422043561935425),
                        ('teratology', 0.4421703815460205), ('mammalian', 0.44115662574768066),
                        ('Nancy_Tullos', 0.44104403257369995), ('opposable_thumb', 0.44093167781829834),
                        ('multi_celled_organisms', 0.4391889274120331), ('nonhuman_primate', 0.4391160309314728),
                        ('interferon_beta_protein', 0.43862149119377136), ('superintelligent', 0.43735820055007935),
                        ('develops_allosteric_modulators', 0.4372589886188507),
                        ('platform_Platinum_HRM', 0.43396157026290894), ('chemosignals', 0.43385976552963257),
                        ('Marilyn_Hausammann', 0.4331941604614258), ('evolutionarily_adaptive', 0.43287330865859985),
                        ('skeletons_skulls', 0.4327971339225769), ('evolutionarily_ancient', 0.43268346786499023),
                        ('fossilized_insides', 0.43256956338882446), ('Kenexa_Nasdaq_KNXA', 0.4306950569152832),
                        ('Frank_Ashen', 0.43065547943115234), ('Elizabeth_Billmeyer', 0.43019911646842957),
                        ('embryological', 0.4297667443752289), ('evolutionary_lineages', 0.42965102195739746),
                        ('sentient_creatures', 0.42872700095176697), ('John_McDarment', 0.4279573857784271),
                        ('notochord', 0.4274761974811554), ('coordinator_Setareh_Yavari', 0.4273516535758972),
                        ('inter_relatedness', 0.4271700382232666), ('Simonetta_Di_Pipo', 0.42710989713668823),
                        ('innovative_microdroplet_based', 0.42686283588409424),
                        ('evolutionary_origins', 0.4258582890033722), ('perfectibility', 0.4251110851764679),
                        ('infinitely_malleable', 0.42477861046791077), ('morally_licit', 0.4243515729904175),
                        ('biogenetic', 0.42430567741394043), ('Dr._Maria_Neira', 0.4240368902683258),
                        ('transhuman', 0.4237884283065796)]

        self.assertEqual(res, expected_res)

        t = time.process_time()
        res = embedding_space.gen_similar_words('human', embedding_space.word2vec)
        elapsed_time_2 = time.process_time() - t

        expected_res = [('human_beings', 0.613968014717102), ('humans', 0.5917960405349731),
                        ('impertinent_flamboyant_endearingly', 0.5868302583694458),
                        ('employee_Laura_Althouse', 0.5639358758926392), ('humankind', 0.5636305809020996),
                        ('Human', 0.5524993538856506), ('mankind', 0.5346406698226929),
                        ('Christine_Gaugler_head', 0.5272535681724548), ('humanity', 0.5262271165847778),
                        ('sentient_intelligent', 0.5201493501663208), ('nonhuman', 0.5158316493034363),
                        ('nonhuman_animals', 0.5148903131484985), ('growth_hormone_misbranding', 0.5118885040283203),
                        ('nonhuman_species', 0.5054227113723755), ('tiniest_fragments', 0.5019494891166687),
                        ('animal', 0.4987776577472687), ('Treating_Gina', 0.4941716492176056),
                        ('Broadcastr_seeks', 0.4918348491191864), ('Stephen_Mimnaugh', 0.4915102422237396),
                        ('sentient_beings', 0.4901559352874756), ('bodily', 0.48965349793434143),
                        ('nonhumans', 0.48545318841934204), ('Douglas_Interiano_spokesman', 0.4853293299674988),
                        ('intelligence_HUMINT', 0.48333364725112915), ('®_Nasdaq_KNXA', 0.4819592237472534),
                        ('machine_interfaces_HMIs', 0.48134148120880127),
                        ('evolutionary_adaptations', 0.47514891624450684), ('interfaces_HMI', 0.47141575813293457),
                        ('homo_sapiens', 0.46880772709846497), ('beings', 0.4686187505722046),
                        ('Human_beings', 0.46829017996788025), ('mankinds', 0.4668985605239868),
                        ('finiteness', 0.4645306468009949), ('executive_Nancy_Tullos', 0.4604729413986206),
                        ('inanimate', 0.45744433999061584), ('prehuman', 0.4573291540145874),
                        ('transmissable', 0.4564620852470398), ('embryo_clones', 0.45624130964279175),
                        ('fallenness', 0.4560883939266205), ('addition_Exponent_evaluates', 0.45543545484542847),
                        ('sentient_creature', 0.45398035645484924), ('sentient', 0.45397087931632996),
                        ('bullet_shell_casing', 0.4528302550315857), ('Humans', 0.4526465833187103),
                        ('Addex_Pharmaceuticals_www.addexpharma.com_discovers', 0.45002347230911255),
                        ('John_L._VandeBerg', 0.44979870319366455), ('natural', 0.44833338260650635),
                        ('displaying_depraved_indifference', 0.4483269453048706), ('humanness', 0.4478735327720642),
                        ('sentience', 0.44722604751586914), ('Savvion_BPM_platform', 0.4471215009689331),
                        ('superorganism', 0.4470677971839905), ('NASDAQ_KNXA_global', 0.4459344148635864),
                        ('vertebrate_animals', 0.445374071598053), ('nonliving', 0.44524049758911133),
                        ('spiritual_beings', 0.44508519768714905), ('comparative_anatomy', 0.44482192397117615),
                        ('creaturely', 0.44442033767700195), ('chimp_genomes', 0.4440094232559204),
                        ('Patricia_Prue', 0.4430835247039795), ('mortal_beings', 0.4427467882633209),
                        ('finitude', 0.44232380390167236), ('bipedal_locomotion', 0.4422861635684967),
                        ('totipotent', 0.44224387407302856), ('animality', 0.4422043561935425),
                        ('teratology', 0.4421703815460205), ('mammalian', 0.44115662574768066),
                        ('Nancy_Tullos', 0.44104403257369995), ('opposable_thumb', 0.44093167781829834),
                        ('multi_celled_organisms', 0.4391889274120331), ('nonhuman_primate', 0.4391160309314728),
                        ('interferon_beta_protein', 0.43862149119377136), ('superintelligent', 0.43735820055007935),
                        ('develops_allosteric_modulators', 0.4372589886188507),
                        ('platform_Platinum_HRM', 0.43396157026290894), ('chemosignals', 0.43385976552963257),
                        ('Marilyn_Hausammann', 0.4331941604614258), ('evolutionarily_adaptive', 0.43287330865859985),
                        ('skeletons_skulls', 0.4327971339225769), ('evolutionarily_ancient', 0.43268346786499023),
                        ('fossilized_insides', 0.43256956338882446), ('Kenexa_Nasdaq_KNXA', 0.4306950569152832),
                        ('Frank_Ashen', 0.43065547943115234), ('Elizabeth_Billmeyer', 0.43019911646842957),
                        ('embryological', 0.4297667443752289), ('evolutionary_lineages', 0.42965102195739746),
                        ('sentient_creatures', 0.42872700095176697), ('John_McDarment', 0.4279573857784271),
                        ('notochord', 0.4274761974811554), ('coordinator_Setareh_Yavari', 0.4273516535758972),
                        ('inter_relatedness', 0.4271700382232666), ('Simonetta_Di_Pipo', 0.42710989713668823),
                        ('innovative_microdroplet_based', 0.42686283588409424),
                        ('evolutionary_origins', 0.4258582890033722), ('perfectibility', 0.4251110851764679),
                        ('infinitely_malleable', 0.42477861046791077), ('morally_licit', 0.4243515729904175),
                        ('biogenetic', 0.42430567741394043), ('Dr._Maria_Neira', 0.4240368902683258),
                        ('transhuman', 0.4237884283065796)]

        self.assertEqual(res, expected_res)

        self.assertLess(elapsed_time_2, 1)
        self.assertLess(elapsed_time_2, elapsed_time_1)

    def test_check_if_unigram_should_be_added(self):
        embedding_space = EmbeddingSpace(self.parameters)
        embedding_space.initialize()
        res = embedding_space.check_if_unigram_should_be_added("planemaker", 0.75, [], "Airbus")
        self.assertEqual(res, True)
