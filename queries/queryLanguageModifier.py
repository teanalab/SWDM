import os
import re
import sys

from bs4 import BeautifulSoup
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import KFold

from embeddings.embedding_space import EmbeddingSpace
from parameters.parameters import Parameters
from sdm.expanded_sdm import ExpandedSdm

sys.path.insert(0, os.path.abspath('..'))
try:
    from runs.runs import Runs
    from queries.queries import Queries
except:
    raise

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class QueryLanguageModifier(object):
    def __init__(self, parameters):
        self.parameters = parameters
        self.embedding_space = EmbeddingSpace(self.parameters)
        self.expanded_sdm = ExpandedSdm(self.parameters)

    @staticmethod
    def _find_all_queries(soup):
        queries = soup.findAll("query")
        return queries

    def _get_query_numbers_to_keep(self, queries, is_test):

        query_numbers = [q.find("number").string for q in queries]

        number_of_folds = int(self.parameters.params["cross_validation"]["number_of_folds"])
        testing_fold = int(self.parameters.params["cross_validation"]["testing_fold"])

        kf = KFold(n_splits=number_of_folds, random_state=0, shuffle=True)

        all_kf_indices = list(kf.split(query_numbers))

        if is_test:
            return [query_numbers[i] for i in all_kf_indices[testing_fold][1]]
        else:
            return [query_numbers[i] for i in all_kf_indices[testing_fold][0]]

    @staticmethod
    def _keep_cv_queries(queries, query_numbers_to_keep):
        for q in queries:
            if q.find("number").string not in query_numbers_to_keep:
                q.decompose()

    @staticmethod
    def _find_unigrams_original(text):
        tknzr = TweetTokenizer()
        return tknzr.tokenize(text)

    def _gen_sdm_fields_texts(self, text):
        sdm_fields_texts = dict()
        self.expanded_sdm.init_top_docs_run_query(text)
        unigrams_original = self._find_unigrams_original(text)
        unigrams_in_embedding_space = self.embedding_space.find_unigrams_in_embedding_space(unigrams_original)
        sdm_fields_texts['u'] = self.expanded_sdm.gen_sdm_field_1_text(unigrams_in_embedding_space, "#combine")
        sdm_fields_texts['o'] = self.expanded_sdm.gen_sdm_field_1_text(unigrams_in_embedding_space, "#od")
        sdm_fields_texts['w'] = self.expanded_sdm.gen_sdm_field_1_text(unigrams_in_embedding_space, "#uw")
        return sdm_fields_texts

    @staticmethod
    def _gen_weighted_fields_text(field_weights, field_texts):
        new_q_text = "\n#weight(\n"
        for field_name, field_weight in field_weights.items():
            field_weight = '{0:.5f}'.format(field_weight)
            q_text = field_texts.get(field_name)
            if float(field_weight) > 0 and len(q_text) > 14:
                combine_text = field_weight + q_text
                new_q_text += combine_text
        new_q_text += ")\n"
        return new_q_text

    def _update_queries(self, queries, field_weights):
        for q in queries:
            if str(q) != "<None></None>":
                q_text = q.find("text")
                q_text_ = q_text.text.strip()
                q_text_ = re.sub('[^0-9a-zA-Z]+', ' ', q_text_)
                field_texts = self._gen_sdm_fields_texts(q_text_)
                q_text.string = self._gen_weighted_fields_text(field_weights, field_texts)

    @staticmethod
    def _post_process_indri_run_query_cfg(soup_str):
        soup_str = soup_str.replace("<trecformat>", "<trecFormat>").replace("</trecformat>", "</trecFormat>")
        soup_str = soup_str.replace("<printquery>", "<printQuery>").replace("</printquery>", "</printQuery>")
        soup_str = soup_str.replace("<fbdocs>", "<fbDocs>").replace("</fbdocs>", "</fbDocs>")
        soup_str = soup_str.replace("<fbterms>", "<fbTerms>").replace("</fbterms>", "</fbTerms>")
        soup_str = soup_str.replace("</workingSetDocno>", "</workingSetDocno>\n")
        return soup_str

    def _update_indri_query_file(self, soup, new_indri_query_file):
        with open(new_indri_query_file, 'w') as f:
            soup_str = str(soup.body.parameters)
            soup_str = self._post_process_indri_run_query_cfg(soup_str)
            print(soup_str, file=f)

    @staticmethod
    def _update_index_dir(soup, index_dir):
        index = soup.find('index')
        index.string = index_dir

    @staticmethod
    def _get_runs_for_re_rank(previous_runs_file):
        if previous_runs_file is None:
            return None
        else:
            return Runs().runs_file_to_documents_dict(previous_runs_file)

    @staticmethod
    def _update_relevance_feedback(soup, fb_terms, fb_docs):
        if fb_terms > 0 and fb_docs > 0:
            soup_parameters = soup.find("parameters")
            soup_tmp = BeautifulSoup("", "lxml")
            fb_terms_tag = soup_tmp.new_tag('fbTerms')
            fb_terms_tag.string = str(fb_terms)
            soup_parameters.append(fb_terms_tag)
            fb_docs_tag = soup_tmp.new_tag('fbDocs')
            fb_docs_tag.string = str(fb_docs)
            soup_parameters.append(fb_docs_tag)

    @staticmethod
    def _remove_empty_queries(queries):
        for q in queries:
            if q.find("text") is not None and q.find("text").text.strip() == "#weight(\n)":
                q.decompose()

    def run(self, is_test):
        self.embedding_space.initialize()
        self.run_no_word2vec_initialization(is_test)

    def run_no_word2vec_initialization(self, is_test):
        soup = Queries().indri_query_file_2_soup(self.parameters.params["query_files"]["old_indri_query_file"])

        self._update_index_dir(soup, self.parameters.params["repo_dir"])

        queries = self._find_all_queries(soup)

        query_numbers_to_keep = self._get_query_numbers_to_keep(queries, is_test)

        self._keep_cv_queries(queries, query_numbers_to_keep)

        self._update_queries(queries, self.parameters.params["sdm_field_weights"])

        self._remove_empty_queries(queries)

        self._update_relevance_feedback(soup, self.parameters.params["prf"]["fb_terms"],
                                        self.parameters.params["prf"]["fb_docs"])

        self._update_indri_query_file(soup, self.parameters.params["query_files"]["new_indri_query_file"])


if __name__ == "__main__":
    parameters_ = Parameters()
    parameters_.read_from_params_file()
    QueryLanguageModifier(parameters_).run(is_test=True)
