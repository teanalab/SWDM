import glob
import os

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 11 / 16


class RemoveRedirects:
    def __init__(self):
        pass

    @staticmethod
    def read_redirects_en(redirects_en_f):
        redirects_en = dict()
        with open(redirects_en_f, "rt") as fin:
            for line in fin:
                line_splits = line.split('\t')
                redirects_en[line_splits[0]] = line_splits[1].strip()
        return redirects_en

    @staticmethod
    def get_filtered_list_of_files(dir_):
        exp = os.path.join(dir_, 'qrels-*.trectext')
        return glob.glob(exp)

    @staticmethod
    def unique(l):
        res = []
        items = set()
        for i in l:
            item0 = i.split()[0]
            item1 = i.split()[1]
            item2 = i.split()[2]
            if (item0, item1, item2) not in items:
                res += [i]
                items.add((item0, item1, item2))
        return res

    def replace_in_a_file(self, redirects_en, file_name):
        text_list = []
        with open(file_name) as f:
            for line in f:
                line_splits = line.split()
                redirect = redirects_en.get(line_splits[2], "")
                if redirect != "":
                    line = line.replace(line_splits[2], redirect)
                text_list += [line.strip()]
        text_list = self.unique(text_list)
        with open(file_name, "w") as f:
            f.write('\n'.join(text_list))

    def replace_redirects(self, redirects_en, file_names):
        for file_name in file_names:
            self.replace_in_a_file(redirects_en, file_name)

    def run(self):
        redirects_en_f_ = "/scratch/data/dbpedia-2015-10/redirects_en.tsv"
        redirects_en_ = self.read_redirects_en(redirects_en_f_)

        dir_qrels = "../configs/qrels/wikipedia-lod/"
        list_of_files_ = self.get_filtered_list_of_files(dir_qrels)

        self.replace_redirects(redirects_en_, list_of_files_)

    @staticmethod
    def initialization():
        cmd = "cd ../configs/qrels/wikipedia-lod/; "
        cmd += "sed 's/<dbpedia://g' qrels-INEX_LD.txt | sed 's/>//g' > qrels-INEX_LD.trectext && "
        cmd += "sed 's/<dbpedia://g' qrels-SemSearch_ES.txt | sed 's/>//g' > qrels-SemSearch_ES.trectext && "
        cmd += "sed 's/<dbpedia://g' qrels-ListSearch.txt | sed 's/>//g' > qrels-ListSearch.trectext && "
        cmd += "sed 's/<dbpedia://g' qrels-QALD.txt | sed 's/>//g' > qrels-QALD.trectext && "
        cmd += "sed 's/<dbpedia://g' qrels-v3.9.txt | sed 's/>//g' > qrels-v3.9.trectext"
        os.system(cmd)


if __name__ == "__main__":
    removeRedirects = RemoveRedirects()
    removeRedirects.initialization()
    removeRedirects.run()
