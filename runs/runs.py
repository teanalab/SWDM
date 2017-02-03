from __future__ import print_function

import csv
import os
import sys
from collections import defaultdict

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class Runs(object):
    @staticmethod
    def runs_file_to_documents_list(runs_file):
        with open(runs_file, 'r') as f:
            docs = [i[2] for i in list(csv.reader(f, delimiter=' '))]
        return docs

    @staticmethod
    def runs_file_to_documents_dict(runs_file):
        docs = defaultdict(list)
        with open(runs_file, 'r') as f:
            for i in list(csv.reader(f, delimiter=' ')):
                docs[i[0]] += [i[2]]
        return docs

    @staticmethod
    def gen_runs_file_name(query_cfg_file):
        dir_q, file_name = os.path.split(query_cfg_file)
        dir_c, dir_q = os.path.split(dir_q)
        dir_r = os.path.join(dir_c, "runs")
        if not os.path.exists(dir_r):
            print("file: \"" + dir_r + "\" does NOT exist.", file=sys.stderr)
            raise ValueError
        ext = ".dsv"
        file_name_r = os.path.splitext(file_name)[0] + ext
        return os.path.join(dir_r, file_name_r)
