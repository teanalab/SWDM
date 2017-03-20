import sys
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
        topics = infer.infer_topics(doc_text, lda)
        print(topics, file=sys.stderr)
        self.assertEqual(topics, [
            '0.041*"abort" + 0.032*"state" + 0.022*"right" + 0.021*"law" + 0.013*"women" + 0.011*"court" + '
            '0.010*"ban" + 0.009*"constitut" + 0.009*"decis" + 0.008*"legal"',
            '0.047*"militari" + 0.028*"forc" + 0.027*"base" + 0.026*"armi" + 0.018*"air" + 0.016*"command" + '
            '0.016*"veteran" + 0.014*"war" + 0.013*"pentagon" + 0.012*"defens"',
            '0.020*"olymp" + 0.017*"cancer" + 0.015*"breast" + 0.012*"cemeteri" + 0.012*"grave" + 0.011*"remain" + '
            '0.010*"mahonei" + 0.010*"tumor" + 0.009*"north" + 0.009*"buri"',
            '0.098*"school" + 0.057*"student" + 0.027*"univers" + 0.022*"colleg" + 0.020*"teacher" + 0.019*"educ" + '
            '0.016*"high" + 0.011*"class" + 0.010*"parent" + 0.010*"children"',
            '0.028*"newspap" + 0.022*"publish" + 0.020*"new" + 0.019*"book" + 0.014*"magazin" + 0.013*"time" + '
            '0.012*"presid" + 0.012*"editor" + 0.012*"press" + 0.011*"report"',
            '0.039*"solidar" + 0.034*"poland" + 0.021*"polish" + 0.020*"govern" + 0.018*"walesa" + 0.014*"union" + '
            '0.011*"communist" + 0.010*"strike" + 0.009*"talk" + 0.008*"warsaw"',
            '0.026*"trade" + 0.018*"farmer" + 0.017*"farm" + 0.016*"agricultur" + 0.015*"export" + 0.014*"countri" + '
            '0.014*"state" + 0.013*"import" + 0.013*"million" + 0.012*"product"',
            '0.082*"west" + 0.062*"german" + 0.057*"east" + 0.051*"germani" + 0.019*"berlin" + 0.012*"border" + '
            '0.011*"hungari" + 0.009*"offici" + 0.009*"kohl" + 0.006*"austria"',
            '0.028*"south" + 0.023*"govern" + 0.020*"africa" + 0.016*"nation" + 0.014*"african" + 0.012*"unit" + '
            '0.011*"state" + 0.010*"cuban" + 0.009*"countri" + 0.008*"presid"',
            '0.032*"vietnam" + 0.023*"vietnames" + 0.018*"cambodia" + 0.017*"roug" + 0.016*"khmer" + '
            '0.014*"refuge" + 0.014*"govern" + 0.011*"cambodian" + 0.010*"thailand" + 0.009*"offici"',
            '0.121*"prison" + 0.039*"sentenc" + 0.029*"inmat" + 0.023*"jail" + 0.023*"convict" + '
            '0.021*"releas" + 0.015*"serv" + 0.012*"year" + 0.012*"escap" + 0.011*"state"',
            '0.028*"republ" + 0.014*"yugoslavia" + 0.011*"haiti" + 0.011*"ethnic" + 0.011*"govern" + '
            '0.008*"kosovo" + 0.007*"yugoslav" + 0.007*"gregg" + 0.006*"haitian" + 0.006*"provinc"',
            '0.028*"north" + 0.017*"document" + 0.016*"contra" + 0.012*"iran" + 0.011*"secur" + 0.011*"former" + '
            '0.010*"reagan" + 0.010*"trial" + 0.009*"aid" + 0.008*"inform"',
            '0.024*"mexico" + 0.017*"presid" + 0.016*"perez" + 0.014*"garcia" + 0.012*"madrid" + 0.012*"carlo" + '
            '0.011*"salina" + 0.011*"venezuela" + 0.010*"rodriguez" + 0.010*"spanish"',
            '0.022*"oil" + 0.020*"train" + 0.015*"spill" + 0.012*"accid" + 0.012*"car" + 0.012*"driver" + 0.011*'
            '"exxon" + 0.010*"alaska" + 0.010*"offici" + 0.009*"railroad"'])
