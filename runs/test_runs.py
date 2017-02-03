from __future__ import print_function

from unittest import TestCase

from runs import Runs

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 23 / 16


class TestRuns(TestCase):
    def setUp(self):
        pass

    def test_runs_file_to_documents_dict(self):
        runs_file = "test_files/runs.dsv"
        res = Runs().runs_file_to_documents_dict(runs_file)
        expected_res = {'TREC_Entity-20': ['David_Zeidler', 'Late_Call_(TV_programme)', 'Alex_Castles',
                                           'The_Great_Moonshine_Conspiracy_Trial_of_1935', 'Buddy_%26_Soul',
                                           '22nd_Legislative_District_(New_Jersey)', 'Pickapeppa_sauce'],
                        'SemSearch_LS-1': ['Buzz_Aldrin', 'Fred_Haise', 'Lost_Moon', 'Apollo_11_goodwill_messages',
                                           'Lunar_Landing_Research_Facility', 'Lunar_Roving_Vehicle', 'Edgar_Mitchell',
                                           'ILC_Dover'],
                        'INEX_LD-20120112': ['United_States_Ambassador_to_Vietnam',
                                             'Distinguished_Service_Order_(Vietnam)',
                                             'Diane_Carlson_Evans', 'Tr%E1%BA%A7n_Hanh', 'Early-arriving_fact',
                                             'II_Field_Force,_Vietnam'],
                        'INEX_LD-20120111': ['List_of_movie_television_channels', 'R-Point',
                                             'Awards_and_decorations_of_the_Vietnam_War',
                                             'Terminology_of_the_Vietnam_War',
                                             'Military_Assistance_Command,_Vietnam_%E2%80%93_Studies_and_Observations' +
                                             '_Group'],
                        'SemSearch_ES-129': ['WXYT-FM', 'Laura_Freele_Osborn', 'Vincent_M._Brennan',
                                             'List_of_Detroit_Lions_players', 'Detroit_Steel', 'Saul_Green',
                                             'MGM_Grand_Detroit', 'Glen_Skov']}
        self.assertEqual(res, expected_res)

    def test_gen_runs_file_name(self):
        query_cfg_file = "../configs/queries/robust04.cfg"
        res = Runs().gen_runs_file_name(query_cfg_file)
        expected_res = "../configs/runs/robust04.dsv"
        self.assertEqual(res, expected_res)
