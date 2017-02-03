from bs4 import BeautifulSoup


class Queries(object):
    def __init__(self):
        pass

    @staticmethod
    def indri_query_file_2_soup(indri_query_file):
        with open(indri_query_file, 'r') as f:
            indri_query_file_s = f.read()
        soup = BeautifulSoup(indri_query_file_s, "lxml")
        return soup
