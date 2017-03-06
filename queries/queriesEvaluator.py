from __future__ import print_function

import csv
import os
import re
import subprocess
from collections import defaultdict

import sys

from parameters.parameters import Parameters
from runs.runs import Runs

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class QueriesEvaluator(object):
    def __init__(self, parameters):
        self.parameters = parameters
        self.evals_d = defaultdict(dict)
        pass

    @staticmethod
    def write_runs_to_file(runs, run_file):
        with open(run_file, 'wb') as f:
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
    def check_indri_run_query_exception(runs_file_name):
        with open(runs_file_name, "r") as f:
            exception_lines = [line for line in f if re.search('EXCEPTION', line)]
            print(exception_lines, file=sys.stderr)
            raise Exception('The IndriRunQuery returned an EXCEPTION')

    @staticmethod
    def write_evals_to_files(evals, eval_file_name):
        with open(eval_file_name, 'wb') as f:
            f.write(evals)

    def evals_file_to_dict(self, eval_file_name):
        self.evals_d.clear()
        with open(eval_file_name, "rt", encoding="ascii") as f:
            evals = list(csv.reader(f, delimiter='\t'))
            evals = [[i.strip() for i in e] for e in evals]
            for (measure, topic, value) in evals:
                self.evals_d[measure][topic] = value

    def run(self):

        indrirunquery_bin = self.parameters.params["indrirunquery_bin"]
        query_cfg_file = self.parameters.params["query_files"]["new_indri_query_file"]
        trec_eval_bin = self.parameters.params["evaluation"]["trec_eval_bin"]
        measure = self.parameters.params["evaluation"]["measure"]
        qrels_file_name = self.parameters.params["evaluation"]["qrels_file_name"]

        runs = self.run_query(indrirunquery_bin, query_cfg_file)
        run_file_name = Runs().gen_runs_file_name(query_cfg_file)
        self.write_runs_to_file(runs, run_file_name)

        self.check_indri_run_query_exception(run_file_name)

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
    parameters_ = Parameters()
    parameters_.read_from_params_file()
    eval_res = QueriesEvaluator(parameters_).run()
    print(eval_res)
