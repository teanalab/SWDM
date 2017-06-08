import csv
from collections import defaultdict

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class Qrels(object):
    def __int__(self):
        pass

    @staticmethod
    def file_2_dict(qrels_file_name):
        l = defaultdict(list)
        with open(qrels_file_name, 'r') as f:
            for i in list(csv.reader(f, delimiter='\t')):
                if i:
                    l[i[0]] += [i[2]]
        return l
