import csv
from collections import defaultdict


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

    @staticmethod
    def gen_qrels_file_name(query_set_opt):
        available_query_set_names = ["ListSearch", "QALD", "INEX_LD", "SemSearch_ES", "v3.9"]
        if query_set_opt not in available_query_set_names:
            raise ValueError(query_set_opt + " not in " + str(available_query_set_names))
        return "../configs/qrels/wikipedia-lod/qrels-" + query_set_opt + ".trectext"
