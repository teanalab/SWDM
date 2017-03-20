from unittest import TestCase

import gensim

from lda.infer import Infer
from parameters.parameters import Parameters


class TestInfer(TestCase):

    def setUp(self):
        self.parameters = Parameters()
        self.parameters.params["repo_dir"] = '/scratch/index/indri_5_7/ap8889'
        self.parameters.params["lda"] = {"num_topics": 100, "file_name": "../configs/lda/ap8889.txt",
                                         "model": "../configs/lda/ap8889.model",
                                         "corpus": "../configs/lda/ap8889.corpus"}

    def test_infer_topics(self):
        infer = Infer(self.parameters)
        infer.add_corpus()
        doc_text = """   More than 150 former officers of the
overthrown South Vietnamese government have been released from a
re-education camp after 13 years of detention, the official Vietnam
News Agency reported Saturday.
   The report from Hanoi, monitored in Bangkok, did not give
specific figures, but said those freed Friday included an
ex-Cabinet minister, a deputy minister, 10 generals, 115
field-grade officers and 25 chaplains.
   It quoted Col. Luu Van Ham, director of the Nam Ha camp south of
Hanoi, as saying all 700 former South Vietnamese officials who had
been held at the camp now have been released.
   They were among 1,014 South Vietnamese who were to be released
from re-education camps under an amnesty announced by the Communist
government to mark Tet, the lunar new year that begins Feb. 17.
   The Vietnam News Agency report said many foreign journalists and
a delegation from the Australia-Vietnam Friendship Association
attended the Nam Ha release ceremony.
   It said Lt. Gen. Nguyen Vinh Nghi, former commander of South
Vietnam's Third Army Corps, and Col. Tran Duc Minh, former director
of the Army Infantry Officers School, expressed ``gratitude to the
government for its humane treatment in spite of the fact that most
of them (detainees) had committed heinous crimes against the
country and people.''
   The prisoners had been held without formal charges or trial
since North Vietnam defeated the U.S.-backed South Vietnamese
government in April 1975, ending the Vietnam War.
   Communist authorities had called the prisoners war criminals and
said they had to learn how to become citizens of the new society.
   Small numbers had been released occasionally without publicity
but the government announced last year that 480 political prisoners
would be freed to mark National Day on Sept. 2.
   On Thursday, Vice Minister of Information Phan Quang said 1,014
would be released under the Tet amnesty.
   He reported a total of 150 prisoners would remain in the camps,
which he said once held 100,000.
   ``Depending on their repentance, they will gradually be released
within a short period of time,'' Quang said.
   He said many of the former inmates would return to their
families in Ho Chi Minh City, formerly the South Vietnamese capital
of Saigon.
   The amnesties apparently are part of efforts by Communist Party
chief Nguyen Van Linh to heal internal divisions and improve
Vietnam's image abroad.
"""
        lda = gensim.models.ldamodel.LdaModel.load(self.parameters.params["lda"]["model"])
        infer.infer_topics(doc_text, lda)
