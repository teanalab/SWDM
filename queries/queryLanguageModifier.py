from __future__ import print_function

import os
import sys

from bs4 import BeautifulSoup

from queries import Queries

sys.path.insert(0, os.path.abspath('..'))
try:
    from runs.runs import Runs
except:
    raise

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class QueryLanguageModifier(object):
    def __init__(self):
        pass

    @staticmethod
    def find_all_queries(soup):
        queries = soup.findAll("query")
        return queries

    @staticmethod
    def gen_combine_1_field(field_weight, field_text):
        return " " * 6 + str(field_weight) + field_text + "\n"

    def gen_combine_fields_text(self, field_weights, field_texts):
        new_q_text = "\n" + " " * 3 + "#weight(\n"
        for field_name, field_weight in field_weights.iteritems():
            q_text = field_texts.get(field_name)
            combine_text = self.gen_combine_1_field(field_weight, q_text)
            new_q_text += combine_text
        new_q_text += " " * 3 + ")\n"
        return new_q_text

    @staticmethod
    def get_bigrams_list(text):
        text_l = text.split(' ')
        bigrams_l = []
        for i in range(0, len(text_l)-1):
            bigrams_l += [text_l[i] + " " + text_l[i+1]]
        return bigrams_l

    def compute_weight(self, gram):
        pass

    def gen_sdm_field_1_text(self, grams, operator):
        sdm_field_text = "#weight(\n"
        for gram in grams:
            sdm_field_text += str(self.compute_weight(gram)) + \
                          " " + operator + "(" + gram + ")\n"
        sdm_field_text += ")\n"
        return sdm_field_text

    def gen_sdm_fields_texts(self, text):
        sdm_fields_texts = dict()
        bigrams_l = self.get_bigrams_list(text)
        sdm_fields_texts['u'] = self.gen_sdm_field_1_text(text.split(" "), "#combine")
        sdm_fields_texts['o'] = self.gen_sdm_field_1_text(bigrams_l, "#od4")
        sdm_fields_texts['w'] = self.gen_sdm_field_1_text(bigrams_l, "#uw17")
        return sdm_fields_texts

    def update_queries(self, queries, field_weights):
        for q in queries:
            q_text = q.find("text")
            field_texts = self.gen_sdm_fields_texts(q_text.text)
            q_text.string = self.gen_combine_fields_text(field_weights, field_texts)

    @staticmethod
    def post_process_indri_run_query_cfg(soup_str):
        soup_str = soup_str.replace("<trecformat>", "<trecFormat>").replace("</trecformat>", "</trecFormat>")
        soup_str = soup_str.replace("<printquery>", "<printQuery>").replace("</printquery>", "</printQuery>")
        soup_str = soup_str.replace("<fbdocs>", "<fbDocs>").replace("</fbdocs>", "</fbDocs>")
        soup_str = soup_str.replace("<fbterms>", "<fbTerms>").replace("</fbterms>", "</fbTerms>")
        soup_str = soup_str.replace("</workingSetDocno>", "</workingSetDocno>\n")
        return soup_str

    def update_indri_query_file(self, soup, new_indri_query_file):
        with open(new_indri_query_file, 'w') as f:
            soup_str = str(soup.body.parameters)
            soup_str = self.post_process_indri_run_query_cfg(soup_str)
            print(soup_str, file=f)

    @staticmethod
    def update_index_dir(soup, index_dir):
        index = soup.find('index')
        index.string = index_dir

    @staticmethod
    def get_runs_for_re_rank(previous_runs_file):
        if previous_runs_file is None:
            return None
        else:
            return Runs().runs_file_to_documents_dict(previous_runs_file)

    @staticmethod
    def update_relevance_feedback(soup, fb_terms, fb_docs):
        if fb_terms > 0 and fb_docs > 0:
            soup_parameters = soup.find("parameters")
            soup_tmp = BeautifulSoup("", "lxml")
            fb_terms_tag = soup_tmp.new_tag('fbTerms')
            fb_terms_tag.string = str(fb_terms)
            soup_parameters.append(fb_terms_tag)
            fb_docs_tag = soup_tmp.new_tag('fbDocs')
            fb_docs_tag.string = str(fb_docs)
            soup_parameters.append(fb_docs_tag)

    def run(self, old_indri_query_file, new_indri_query_file, index_dir, field_weights, fb_terms,
            fb_docs):

        soup = Queries().indri_query_file_2_soup(old_indri_query_file)

        self.update_index_dir(soup, index_dir)

        queries = self.find_all_queries(soup)
        self.update_queries(queries, field_weights)

        self.update_relevance_feedback(soup, fb_terms, fb_docs)

        self.update_indri_query_file(soup, new_indri_query_file)


if __name__ == "__main__":
    field_weights_ = {
        "u": 0.8,
        "o": 0.1,
        "w": 0.1
    }
    index_dir_ = "/scratch/index/indri_5_7/robust/"
    old_indri_query_file_ = "../configs/queries/robust04.cfg"
    new_indri_query_file_ = "../configs/queries/robust04_expanded.cfg"
    fb_docs_ = 10
    fb_terms_ = 10
    QueryLanguageModifier().run(old_indri_query_file_, new_indri_query_file_, index_dir_, field_weights_,
                                fb_terms_, fb_docs_)
