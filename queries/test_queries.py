from unittest import TestCase

from queries.queries import Queries

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class TestQueries(TestCase):
    def test_indri_query_file_2_soup(self):
        indri_query_file = 'test_files/indri_query.cfg'
        soup = Queries().indri_query_file_2_soup(indri_query_file)

        expected_res = "<query>\n" \
                       "<number>INEX_LD-20120121</number>\n" \
                       "<text>vietnam food recipes</text>\n" \
                       "</query>"
        res = soup.find("number", text="INEX_LD-20120121").parent
        self.assertEqual(str(res), expected_res)

        self.assertEqual(soup.find("index").text, "/scratch/index/indri_5_7/dbpedia_2015_10_short_abstracts_en/")
        self.assertEqual(soup.findAll("query")[0].find("number").text, "INEX_LD-20120111")
