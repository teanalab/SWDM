from __future__ import print_function

from unittest import TestCase

from qrels.qrels import Qrels


class TestQrels(TestCase):
    def test_file_2_dict(self):
        res = Qrels().file_2_dict('test_files/qrels.trectext')
        expected_res = {'SemSearch_ES-1': ['.444_Marlin', '.41_Remington_Magnum'],
                        'SemSearch_ES-102': ['Camissoniopsis_cheiranthifolia', 'Calystegia_soldanella'],
                        'SemSearch_ES-10': ['Propheteer', 'Tom_Bradley_(baseball)',
                                            'Asheville_metropolitan_area', 'University_of_North_Carolina',
                                            'Ayden,_North_Carolina', 'Ashville',
                                            'University_of_North_Carolina_at_Asheville', 'U.S._Route_321',
                                            'Asheville,_North_Carolina', '2008_North_Carolina_Tar_Heels_football_team',
                                            'Clyde,_North_Carolina'], 'SemSearch_ES-100': ['Tampa,_Florida', 'YMCA'],
                        'SemSearch_ES-101': ['Ashley_Wagner']}
        self.assertEqual(res, expected_res)
