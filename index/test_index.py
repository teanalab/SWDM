import sys
from unittest import TestCase

import index.index
from collection.simple_document import SimpleDocument
from parameters.parameters import Parameters


class TestIndex(TestCase):
    def setUp(self):
        self.parameters = Parameters()
        self.parameters.params["repo_dir"] = '../index/test_files/index'

        self.index_ = index.index.Index(self.parameters)

    def test_uw_expression_count(self):
        self.assertEqual(self.index_.uw_expression_count("SAMPSON Dog", 12), 2)

    def test_od_expression_count(self):
        self.assertEqual(self.index_.od_expression_count("SAMPSON True", 12), 1)

    def test_uw_document_expression_count(self):
        self.assertEqual(self.index_.uw_expression_document_count("SAMPSON True", 12), 1)

    def test_od_document_expression_count(self):
        self.assertEqual(self.index_.od_expression_document_count("SAMPSON True", 12), 1)

    def test_term_count(self):
        self.assertEqual(self.index_.term_count("dog"), 2)

        self.parameters.params["repo_dir"] = '/scratch/index/indri_5_7/ap8889'
        self.index_ = index.index.Index(self.parameters)

        self.assertEqual(self.index_.term_count("emotional"), 3515)

    def test_document_count(self):
        self.assertEqual(self.index_.document_count("dog"), 1)

        self.parameters.params["repo_dir"] = '/scratch/index/indri_5_7/ap8889'
        self.index_ = index.index.Index(self.parameters)

        self.assertEqual(self.index_.document_count("emotional"), 2973)

    def test_check_if_have_same_stem(self):
        self.assertEqual(self.index_.check_if_have_same_stem("goes", "goe"), True)
        self.assertEqual(self.index_.check_if_have_same_stem("goes", "g"), False)
        self.assertEqual(self.index_.check_if_have_same_stem("first", "mr"), False)

    def test_idf(self):
        self.assertEqual(self.index_.idf("dog"), 0.4054651081081644)

        self.parameters.params["repo_dir"] = '/scratch/index/indri_5_7/ap8889'
        self.index_ = index.index.Index(self.parameters)

        self.assertEqual(self.index_.document_count("first"), 0)

    def test_tfidf(self):
        self.parameters.params["repo_dir"] = '/scratch/index/indri_5_7/ap8889'

        self.index_ = index.index.Index(self.parameters)
        doc_words = SimpleDocument(self.parameters).get_words("../configs/others/pride_and_prejudice_wiki.txt")
        tfidf_1 = self.index_.tfidf('emotional', doc_words)
        print(tfidf_1, file=sys.stderr)
        tfidf_2 = self.index_.tfidf('is', doc_words)
        print(tfidf_2, file=sys.stderr)

        self.assertEqual(tfidf_1, 0)
        self.assertEqual(tfidf_2, 0)

    def test_tf(self):
        doc_words = SimpleDocument(self.parameters).get_words("../configs/others/pride_and_prejudice_wiki.txt")

        self.assertEqual(self.index_.tf("dog", doc_words), 0.5)

        self.parameters.params["repo_dir"] = '/scratch/index/indri_5_7/ap8889'
        self.index_ = index.index.Index(self.parameters)

        self.assertEqual(self.index_.tf("emotional", doc_words), 0.5096153846153846)

    def test_check_if_exists_in_index(self):
        self.parameters.params["repo_dir"] = '/scratch/index/indri_5_7/ap8889'
        self.index_ = index.index.Index(self.parameters)

        self.assertTrue(self.index_.check_if_exists_in_index("emotional"))
        self.assertFalse(self.index_.check_if_exists_in_index("first"))
        self.assertFalse(self.index_.check_if_exists_in_index("included"))
        self.assertTrue(self.index_.check_if_exists_in_index("includes"))

    def test_obtain_text_of_a_document(self):
        self.parameters.params["repo_dir"] = '/scratch/index/indri_5_7/ap8889'
        self.index_ = index.index.Index(self.parameters)
        res = self.index_.obtain_text_of_a_document(1)
        self.assertEqual(res, """
   Public Order Minister Tassos Sehiotis
resigned Monday after a Greek-American banker indicted in a $30
million financial scandal fled the country, apparently aboard a
yacht.
   The conservative opposition immediately demanded the resignation
of Premier Andreas Papandreou's socialist government, claiming it
was staging a cover-up.
   Banker George Koskotas, 34, disappeared Saturday afternoon. A
police officer, speaking on condition of anonymity, said Koskotas
fled abroad on Sunday, apparently by yacht from the seaside village
of Megalo Pefko, 20 miles from Athens.
   Sehiotis said a warrant had been issued for Koskotas' arrest. One
week ago, Koskotas was banned from leaving Greece pending the
outcome of an official enquiry into alleged financial irregularities
at the Bank of Crete, which he controls.
   Sehiotis, whose ministry was responsible for police surveillance
of Koskotas, said he was resigning ``since such (public order
ministry) omissions ... create an issue of political sensitivity.''
   The scandal has shaken the government because of accusations in
Greek newspapers that senior socialist officials were involved in
illegal deals set up by the Bank of Crete.
   The socialists also have been criticized for permitting Koskotas
to build a multi-million dollar banking and media empire in Greece
since 1984 without adequate checks by the central bank on his
financial background.
   The government last week pledged ``absolute clarity'' in
uncovering the scandal and warned there will be ``no pardons' for
members of the ruling Panhellenic Socialist Movement (PASOK) who may
be implicated.
   ``The Greek people are left with the conviction that George
Koskotas was spirited away so that he would not speak. The
responsibility goes all the way to the top of the government
pyramid,'' Constantine Mitsotakis, leader of the New Democracy main
opposition party party, said in a statement demanding the government
resign.
   Koskotas was suspended Oct. 20 as chairman of the Bank of Crete
and indicted on five counts of forgery and embezzlement.
   Last week Koskotas appeared before a district attorney on a
charge of forging documents purporting to show that the Bank of
Crete had $13 million invested with the American brokerage firm
Merrill Lynch.
   He was not detained but given until Nov. 14 to prepare his
defense.
   Koskotas also is accused of forging documents purporting to show
his bank had another $17 million in an account with an American
bank, Irving Trust Corp. Both U.S. firms have said they had no
record of the deposits.
   Koskotas, who holds both American and Greek citizenship, bought a
controlling interest in the Bank of Crete in 1984 after working in
its central Athens branch for six years as an accountant.
   Rival newspapers have claimed Koskotas illegally used Bank of
Crete money to fund his publishing group Grammi, which controls
three daily newspapers, five magazines and a radio station.
   Koskotas resigned Oct. 29 as chairman of Grammi, the day after
the premier's son, Education Minister George Papandreou, denounced
as a forgery a Bank of Crete statement showing a $2.3 million
transfer to a Merrill Lynch account in his name.
   The younger Papandreou showed reporters a letter from a New York
lawyer saying there was no record at Merrill Lynch of such a
transfer.
   Koskotas' parents, brother, wife and five children all have left
Greece during the past week.
""")

    def test_obtain_term_ids_of_a_document(self):
        self.parameters.params["repo_dir"] = '/scratch/index/indri_5_7/ap8889'
        self.index_ = index.index.Index(self.parameters)
        res = self.index_.obtain_term_ids_of_a_document(1)
        self.assertEqual(res, ('AP881107-0001', (
            147, 771, 0, 78064, 26, 2828, 1283, 92, 126, 147, 175009, 159395, 771, 55, 0, 0, 2362, 26, 2828, 919, 0, 0,
            115,
            8, 461, 1624, 1826, 0, 35, 693, 1198, 0, 195412, 0, 724, 430, 621, 340, 0, 771, 0, 1502, 20649, 4327, 1620,
            9,
            247, 0, 0, 866, 0, 643, 0, 2828, 415, 101374, 1289, 2015, 276, 1246, 0, 24, 29, 586, 0, 272, 0, 856, 0,
            101374,
            1826, 2153, 0, 174, 693, 0, 195412, 0, 0, 158999, 1037, 0, 117013, 137162, 123, 157, 0, 4415, 159395, 0, 0,
            2262, 0, 0, 56, 0, 101374, 251, 0, 0, 189, 101374, 0, 569, 0, 332, 3095, 1873, 0, 2974, 0, 0, 13, 63630, 0,
            485,
            461, 91464, 0, 0, 91, 0, 50405, 0, 0, 156, 159395, 0, 690, 0, 347, 0, 24, 4049, 0, 101374, 0, 0, 0, 771, 0,
            0,
            92, 126, 690, 131907, 609, 0, 56, 0, 88, 2222, 0, 0, 1624, 0, 160974, 0, 9, 0, 0, 436, 0, 2362, 273, 0, 774,
            1620, 13, 0, 263, 0, 887, 339, 176, 0, 0, 0, 91, 0, 50405, 0, 1620, 0, 0, 0, 289, 0, 1202, 101374, 0, 254,
            0,
            4543, 8, 193, 91, 0, 979, 3597, 0, 3095, 0, 791, 0, 2768, 937, 0, 0, 264, 91, 0, 0, 461, 2464, 0, 9, 0, 0,
            1333,
            2198, 45622, 0, 4433, 0, 1624, 0, 678, 0, 0, 0, 0, 3790, 0, 40, 0, 0, 118, 135313, 1620, 727, 136295, 0, 0,
            0,
            3408, 0, 2362, 6, 0, 245, 0, 0, 494, 0, 415, 101374, 0, 2042, 435, 0, 0, 0, 0, 0, 586, 0, 347, 1461, 0, 0,
            116,
            0, 0, 354, 0, 0, 9, 145130, 0, 48310, 120515, 51, 0, 0, 1, 956, 540, 430, 32, 32, 0, 0, 0, 221, 340, 0, 9,
            771,
            101374, 0, 1431, 897, 123, 0, 248, 0, 0, 91, 0, 50405, 0, 919, 0, 114, 667, 0, 70387, 0, 62918, 0, 0,
            101374,
            261, 0, 0, 311, 212, 0, 0, 61, 0, 4322, 767, 144897, 0, 62, 0, 0, 91, 0, 50405, 0, 375, 8, 546, 0, 0, 26,
            3527,
            543, 3657, 3224, 0, 0, 0, 2223, 0, 533, 0, 1019, 367, 0, 674, 0, 165, 101374, 0, 0, 436, 0, 4322, 767,
            144897,
            0, 62, 0, 91, 0, 0, 445, 8, 0, 0, 550, 0, 0, 26, 91, 3533, 1512, 285, 0, 0, 543, 0, 0, 0, 0, 0, 192, 0, 0,
            1798,
            101374, 0, 328, 0, 26, 0, 2362, 45369, 1390, 0, 156, 167, 0, 0, 91, 0, 50405, 0, 791, 0, 28, 0, 0, 264,
            4415,
            1977, 0, 187, 15, 0, 0, 550, 1483, 273, 0, 247, 101374, 887, 0, 91, 0, 50405, 168, 0, 303, 0, 515, 34,
            77704, 0,
            156, 23, 701, 273, 114, 1056, 0, 0, 409, 490, 101374, 771, 897, 899, 0, 248, 0, 77704, 0, 0, 0, 0, 1502,
            599,
            463, 147, 415, 4327, 2512, 0, 0, 70387, 0, 91, 0, 50405, 221, 62, 0, 17, 41, 8, 1548, 0, 0, 3657, 3224, 550,
            0,
            0, 171, 0, 2597, 4327, 62, 4, 0, 512, 0, 0, 1, 73, 532, 7, 0, 0, 0, 192, 0, 3657, 3224, 0, 0, 0, 1548,
            101374,
            626, 918, 455, 0, 114, 219, 0, 0, 245, 3095, 0, 0, 333, 0)))

    def test_obtain_terms_of_a_document(self):
        self.parameters.params["repo_dir"] = '/scratch/index/indri_5_7/ap8889'
        self.index_ = index.index.Index(self.parameters)
        res = self.index_.obtain_terms_of_a_document(1)
        print(res, file=sys.stderr)
        self.assertEqual(res, ('AP881107-0001', (
            'minist', 'resign', '', 'gree', 'american', 'banker', 'escap', 'public', 'order', 'minist', 'tasso',
            'sehioti',
            'resign', 'mondai', '', '', 'greek', 'american', 'banker', 'indict', '', '', '30', 'million', 'financi',
            'scandal', 'fled', '', 'countri', 'appar', 'aboard', '', 'yacht', '', 'conserv', 'opposit', 'immedi',
            'demand',
            '', 'resign', '', 'premier', 'andrea', 'papandr', 'socialist', 'govern', 'claim', '', '', 'stage', '',
            'cover',
            '', 'banker', 'georg', 'koskota', '34', 'disappear', 'saturdai', 'afternoon', '', 'polic', 'offic', 'speak',
            '',
            'condit', '', 'anonym', '', 'koskota', 'fled', 'abroad', '', 'sundai', 'appar', '', 'yacht', '', '',
            'seasid',
            'villag', '', 'megalo', 'pefko', '20', 'mile', '', 'athen', 'sehioti', '', '', 'warrant', '', '', 'issu',
            '',
            'koskota', 'arrest', '', '', 'ago', 'koskota', '', 'ban', '', 'leav', 'greec', 'pend', '', 'outcom', '', '',
            'offici', 'enquiri', '', 'alleg', 'financi', 'irregular', '', '', 'bank', '', 'crete', '', '', 'control',
            'sehioti', '', 'ministri', '', 'respons', '', 'polic', 'surveil', '', 'koskota', '', '', '', 'resign', '',
            '',
            'public', 'order', 'ministri', 'omiss', 'creat', '', 'issu', '', 'polit', 'sensit', '', '', 'scandal', '',
            'shaken', '', 'govern', '', '', 'accus', '', 'greek', 'newspap', '', 'senior', 'socialist', 'offici', '',
            'involv', '', 'illeg', 'deal', 'set', '', '', '', 'bank', '', 'crete', '', 'socialist', '', '', '',
            'critic',
            '', 'permit', 'koskota', '', 'build', '', 'multi', 'million', 'dollar', 'bank', '', 'media', 'empir', '',
            'greec', '', '1984', '', 'adequ', 'check', '', '', 'central', 'bank', '', '', 'financi', 'background', '',
            'govern', '', '', 'pledg', 'absolut', 'clariti', '', 'uncov', '', 'scandal', '', 'warn', '', '', '', '',
            'pardon', '', 'member', '', '', 'rule', 'panhellen', 'socialist', 'movement', 'pasok', '', '', '', 'implic',
            '',
            'greek', 'peopl', '', 'left', '', '', 'convict', '', 'georg', 'koskota', '', 'spirit', 'awai', '', '', '',
            '',
            '', 'speak', '', 'respons', 'goe', '', '', 'wai', '', '', 'top', '', '', 'govern', 'pyramid', '',
            'constantin',
            'mitsotaki', 'leader', '', '', 'new', 'democraci', 'main', 'opposit', 'parti', 'parti', '', '', '',
            'statement',
            'demand', '', 'govern', 'resign', 'koskota', '', 'suspend', 'oct', '20', '', 'chairman', '', '', 'bank', '',
            'crete', '', 'indict', '', 'five', 'count', '', 'forgeri', '', 'embezzl', '', '', 'koskota', 'appear', '',
            '',
            'district', 'attornei', '', '', 'charg', '', 'forg', 'document', 'purport', '', 'show', '', '', 'bank', '',
            'crete', '', '13', 'million', 'invest', '', '', 'american', 'brokerag', 'firm', 'merril', 'lynch', '', '',
            '',
            'detain', '', 'given', '', 'nov', '14', '', 'prepar', '', 'defens', 'koskota', '', '', 'accus', '', 'forg',
            'document', 'purport', '', 'show', '', 'bank', '', '', '17', 'million', '', '', 'account', '', '',
            'american',
            'bank', 'irv', 'trust', 'corp', '', '', 'firm', '', '', '', '', '', 'record', '', '', 'deposit', 'koskota',
            '',
            'hold', '', 'american', '', 'greek', 'citizenship', 'bought', '', 'control', 'interest', '', '', 'bank', '',
            'crete', '', '1984', '', 'work', '', '', 'central', 'athen', 'branch', '', 'six', 'year', '', '', 'account',
            'rival', 'newspap', '', 'claim', 'koskota', 'illeg', '', 'bank', '', 'crete', 'monei', '', 'fund', '',
            'publish', 'group', 'grammi', '', 'control', 'three', 'daili', 'newspap', 'five', 'magazin', '', '',
            'radio',
            'station', 'koskota', 'resign', 'oct', '29', '', 'chairman', '', 'grammi', '', '', '', '', 'premier', 'son',
            'educ', 'minist', 'georg', 'papandr', 'denounc', '', '', 'forgeri', '', 'bank', '', 'crete', 'statement',
            'show', '', '2', '3', 'million', 'transfer', '', '', 'merril', 'lynch', 'account', '', '', 'name', '',
            'younger', 'papandr', 'show', 'report', '', 'letter', '', '', 'new', 'york', 'lawyer', 'sai', '', '', '',
            'record', '', 'merril', 'lynch', '', '', '', 'transfer', 'koskota', 'parent', 'brother', 'wife', '', 'five',
            'children', '', '', 'left', 'greec', '', '', 'past', '')))

    def test_term(self):
        self.parameters.params["repo_dir"] = '/scratch/index/indri_5_7/ap8889'
        self.index_ = index.index.Index(self.parameters)
        res = self.index_.term(147)
        print(res, file=sys.stderr)
        self.assertEqual(res, 'minist')

    def test_expression_list(self):
        self.assertEqual(self.index_.expression_list("SAMPSON Dog", "#uw", 12), {'romeo': 2})
        self.assertEqual(self.index_.expression_list("your", "#uw", 12), {'hamlet': 1, 'romeo': 3})

    def test_run_query(self):
        self.index_.init_query_env()
        self.assertEqual(self.index_.run_query("you"), ((3, -4.207161834249701), (2, -4.27477458192466)))
