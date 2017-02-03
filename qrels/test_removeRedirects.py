import logging
import sys
from unittest import TestCase

from qrels.removeRedirects import RemoveRedirects

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 11 / 16

logger = logging.getLogger()
logger.level = logging.DEBUG
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class TestRemoveRedirects(TestCase):
    def setUp(self):
        self.removeRedirects = RemoveRedirects()
        self.maxDiff = 10000

    def test_uniq(self):
        test = ["INEX_LD-2009022 Q0      History_of_Chinese_cuisine      1",
                "INEX_LD-2009022 Q0      Mapo_doufu      1",
                "INEX_LD-2009022 Q0      Chinatown       1",
                "INEX_LD-2009022 Q0      Hot_and_sour_soup       1",
                "INEX_LD-2009022 Q0      Chinatown       1",
                "INEX_LD-2009022 Q0      Chinatown       2",
                "INEX_LD-2009022 Q0      Pao_cai 1",
                "INEX_LD-2009022 Q0      Shuizhu 1",
                "INEX_LD-2009022 Q0      Guizhou_cuisine 1",
                "INEX_LD-2009022 Q0      Kung_Pao_chicken        1",
                "INEX_LD-2009022 Q0      Chinatown       1",
                "INEX_LD-2009022 Q0      Twice_cooked_pork       1",
                "INEX_LD-2009022 Q0      Shuizhu 1",
                "INEX_LD-2009022 Q0      Vicia_faba      1",
                "INEX_LD-2009022 Q0      Fermentation_in_food_processing 1",
                "INEX_LD-2009022 Q1      Kung_Pao_chicken        1"
                ]

        res = self.removeRedirects.unique(test)

        expected_res = ["INEX_LD-2009022 Q0      History_of_Chinese_cuisine      1",
                        "INEX_LD-2009022 Q0      Mapo_doufu      1",
                        "INEX_LD-2009022 Q0      Chinatown       1",
                        "INEX_LD-2009022 Q0      Hot_and_sour_soup       1",
                        "INEX_LD-2009022 Q0      Pao_cai 1",
                        "INEX_LD-2009022 Q0      Shuizhu 1",
                        "INEX_LD-2009022 Q0      Guizhou_cuisine 1",
                        "INEX_LD-2009022 Q0      Kung_Pao_chicken        1",
                        "INEX_LD-2009022 Q0      Twice_cooked_pork       1",
                        "INEX_LD-2009022 Q0      Vicia_faba      1",
                        "INEX_LD-2009022 Q0      Fermentation_in_food_processing 1",
                        "INEX_LD-2009022 Q1      Kung_Pao_chicken        1"
                        ]

        self.assertEqual(res, expected_res)
