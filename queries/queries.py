from bs4 import BeautifulSoup

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class Queries(object):
    def __init__(self):
        pass

    @staticmethod
    def indri_query_file_2_soup(indri_query_file):
        with open(indri_query_file, 'r') as f:
            indri_query_file_s = f.read()
        soup = BeautifulSoup(indri_query_file_s, "lxml")
        return soup
