from __future__ import print_function

import csv
import os
import subprocess
from collections import defaultdict

from qrels.qrels import Qrels
from queryLanguageModifier import QueryLanguageModifier
from runs.runs import Runs

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class QueriesEvaluator(object):
    def __init__(self):
        self.evals_d = defaultdict(dict)
        pass

    @staticmethod
    def write_runs_to_file(runs, run_file):
        with open(run_file, 'w') as f:
            f.write(runs)

    @staticmethod
    def run_query(indrirunquery_bin, query_cfg_file):
        command_l = [indrirunquery_bin, query_cfg_file]
        command = ' '.join(command_l)
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
        return output

    @staticmethod
    def run_trec_eval_1(trec_eval_bin, run_file_name, qrels_file):
        command_l = [trec_eval_bin, "-q", "-m", "all_trec", qrels_file, run_file_name]
        command = ' '.join(command_l)
        evals = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
        return evals

    @staticmethod
    def write_evals_to_files(evals, eval_file_name):
        with open(eval_file_name, 'w') as f:
            f.write(evals)

    def evals_file_to_dict(self, eval_file_name):
        self.evals_d.clear()
        with open(eval_file_name, 'rb') as f:
            evals = list(csv.reader(f, delimiter='\t'))
            evals = [[i.strip() for i in e] for e in evals]
            for (measure, topic, value) in evals:
                self.evals_d[measure][topic] = value

    def run(self, indrirunquery_bin, trec_eval_bin, query_cfg_file, measure, query_set_opt):

        runs = self.run_query(indrirunquery_bin, query_cfg_file)
        run_file_name = Runs().gen_runs_file_name(query_cfg_file)
        self.write_runs_to_file(runs, run_file_name)

        qrels_file_name = Qrels().gen_qrels_file_name(query_set_opt)
        evals = self.run_trec_eval_1(trec_eval_bin, run_file_name, qrels_file_name)
        eval_file_name = self.gen_configs_file_name(query_cfg_file, "evals", ".dsv")
        self.write_evals_to_files(evals, eval_file_name)

        self.evals_file_to_dict(eval_file_name)
        return self.evals_d[measure]['all']

    def get_all_measures(self):
        return self.evals_d

    @staticmethod
    def gen_configs_file_name(query_cfg_file, configs_dir, ext):
        dir_q, file_name = os.path.split(query_cfg_file)
        dir_c, dir_q = os.path.split(dir_q)
        dir_r = os.path.join(dir_c, configs_dir)
        if not os.path.exists(dir_r):
            raise ValueError("file " + dir_r + " does NOT exist.")
        file_name_r = os.path.splitext(file_name)[0] + ext
        return os.path.join(dir_r, file_name_r)


if __name__ == "__main__":
    query_cfg_file_ = "../configs/queries/robust04_expanded.cfg"
    params_m_ = {"index_dir": "/scratch/index/indri_5_7/robust/",
                 "old_indri_query_file": "../configs/queries/robust04.cfg",
                 "new_indri_query_file": query_cfg_file_,
                 "previous_runs_file": "../configs/runs/LOD_short_abstracts_for_image_extraction.dsv",
                 "field_weights": {"u": 0.8, "o": 0.1, "w": 0.1}}
    QueryLanguageModifier().run(**params_m_)

    indrirunquery_bin_ = "/home/fj9124/thirdPartyProgs/indri-5.11/build/bin/IndriRunQuery"
    trec_eval_bin_ = "/home/fj9124/projects/ir/evaluators/trec_eval/trec_eval.9.0/trec_eval"
    measure_ = 'map'
    query_set_opt_ = 'SemSearch_ES'
    eval_res = QueriesEvaluator().run(indrirunquery_bin_, trec_eval_bin_, query_cfg_file_, measure_, query_set_opt_)
    print(eval_res)
