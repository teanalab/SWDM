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
