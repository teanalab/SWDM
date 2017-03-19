from unittest import TestCase

from lda.train import Train
from parameters.parameters import Parameters


class TestTrain(TestCase):
    def setUp(self):
        self.parameters = Parameters()
        self.parameters.params["repo_dir"] = '/scratch/index/indri_5_7/ap8889'
        self.parameters.params["lda"] = {"num_topics": 100, "file_name": "../configs/lda/ap8889.txt",
                                         "model": "../configs/lda/ap8889.model"}

    def test_run(self):
        train = Train(self.parameters)
        train.add_corpus()
        train.run()
        self.assertEqual(train.lda.print_topics(20),
                         [(69,
                           '0.000*"ideaho" + 0.000*"makati" + 0.000*"rappini" + 0.000*"gillert" + 0.000*"rasa" + '
                           '0.000*"commissariat" + 0.000*"abdcut" + 0.000*"norco" + 0.000*"alibad" + 0.000*"comet"'),
                          (77,
                           '0.000*"polana" + 0.000*"woad" + 0.000*"beween" + 0.000*"quintex" + 0.000*"mutz" + '
                           '0.000*"dalrympl" + 0.000*"bangla" + 0.000*"blusteri" + 0.000*"stefansson" + 0.000*"odgen"'),
                          (94,
                           '0.000*"miso" + 0.000*"kjelmyr" + 0.000*"unsearch" + 0.000*"oleaga" + '
                           '0.000*"disciplinair" + 0.000*"journeywoman" + 0.000*"quicki" + 0.000*"mistat" + '
                           '0.000*"macaluso" + 0.000*"girli"'),
                          (70,
                           '0.000*"papierz" + 0.000*"ginko" + 0.000*"helrich" + 0.000*"kwangu" + 0.000*"hiraga" + '
                           '0.000*"ribl" + 0.000*"acrimoni" + 0.000*"tavel" + 0.000*"undersecratri" + 0.000*"perkin"'),
                          (7,
                           '0.000*"slik" + 0.000*"bubolz" + 0.000*"sneh" + 0.000*"mazraa" + 0.000*"leckband" + '
                           '0.000*"zojoji" + 0.000*"strader" + 0.000*"bizon" + 0.000*"dobbi" + 0.000*"congregr"'),
                          (52,
                           '0.000*"lipnack" + 0.000*"strandberg" + 0.000*"scrimp" + 0.000*"rider" + 0.000*"blackum" + '
                           '0.000*"xenaki" + 0.000*"internacionalista" + 0.000*"gilligan" + 0.000*"raeburn" + '
                           '0.000*"leah"'),
                          (54,
                           '0.000*"benar" + 0.000*"wishart" + 0.000*"tapi" + 0.000*"concilia" + 0.000*"swenei" + '
                           '0.000*"dotur" + 0.000*"schoenk" + 0.000*"hajj" + 0.000*"bernabo" + 0.000*"candida"'),
                          (75,
                           '0.000*"upham" + 0.000*"roedenbeck" + 0.000*"reoccup" + 0.000*"rogelio" + 0.000*"aman" + '
                           '0.000*"tartli" + 0.000*"basauri" + 0.000*"leeon" + 0.000*"igloliort" + 0.000*"ballroom"'),
                          (19,
                           '0.000*"homolog" + 0.000*"longview" + 0.000*"restabil" + 0.000*"stowawai" + 0.000*"ockman" '
                           '+ 0.000*"nardin" + 0.000*"wind" + 0.000*"harn" + 0.000*"safrica" + 0.000*"rosella"'),
                          (29,
                           '0.000*"gorzow" + 0.000*"ashoot" + 0.000*"konold" + 0.000*"dobbert" + 0.000*"untint" + '
                           '0.000*"processs" + 0.000*"wplj" + 0.000*"laitin" + 0.000*"zagrodnik" + 0.000*"ruperez"'),
                          (49,
                           '0.000*"depoli" + 0.000*"antioch" + 0.000*"gulf" + 0.000*"zanini" + 0.000*"machaniam" + '
                           '0.000*"goodl" + 0.000*"eeelam" + 0.000*"audiocassett" + 0.000*"redound" + 0.000*"montril"'),
                          (42,
                           '0.000*"vlasi" + 0.000*"detloff" + 0.000*"tornoto" + 0.000*"vega" + 0.000*"azurin" + '
                           '0.000*"banksid" + 0.000*"saloonkeep" + 0.000*"shropshir" + 0.000*"initiat" + '
                           '0.000*"notion"'),
                          (48,
                           '0.000*"chakiri" + 0.000*"pitirim" + 0.000*"downplai" + 0.000*"clanton" + 0.000*"ungheni" '
                           '+ 0.000*"manfedo" + 0.000*"bisco" + 0.000*"livui" + 0.000*"disinclin" + 0.000*"zapmail"'),
                          (1,
                           '0.000*"tongawalla" + 0.000*"zellerbach" + 0.000*"kinghorn" + 0.000*"tunasan" + '
                           '0.000*"drumfir" + 0.000*"dekruif" + 0.000*"superfici" + 0.000*"echendia" + 0.000*"fuyi" + '
                           '0.000*"comet"'),
                          (97,
                           '0.000*"foreyt" + 0.000*"guigard" + 0.000*"margherita" + 0.000*"cyanimid" + '
                           '0.000*"workerss" + 0.000*"ariagada" + 0.000*"gullei" + 0.000*"lunarscap" + '
                           '0.000*"videira" + 0.000*"camarad"'),
                          (89,
                           '0.000*"rca" + 0.000*"dam" + 0.000*"rago" + 0.000*"rosston" + 0.000*"arguello" + '
                           '0.000*"septeb" + 0.000*"hathem" + 0.000*"videodisc" + 0.000*"knn" + 0.000*"jadovich"'),
                          (76,
                           '0.000*"squirm" + 0.000*"innerspac" + 0.000*"fraggl" + 0.000*"scacchi" + 0.000*"biz" + '
                           '0.000*"hubba" + 0.000*"chirisa" + 0.000*"maricica" + 0.000*"shaalvim" + '
                           '0.000*"peacefulli"'),
                          (41,
                           '0.000*"shewan" + 0.000*"curio" + 0.000*"ishmar" + 0.000*"pandarama" + 0.000*"sherbert" + '
                           '0.000*"mtv" + 0.000*"symphathet" + 0.000*"dhirendra" + 0.000*"sprit" + 0.000*"valuu"'),
                          (56,
                           '0.000*"castellari" + 0.000*"nobl" + 0.000*"ripol" + 0.000*"eastmaqu" + 0.000*"lakeshor" + '
                           '0.000*"hatem" + 0.000*"kavner" + 0.000*"renouard" + 0.000*"riverboat" + 0.000*"meriweth"'),
                          (98,
                           '0.000*"homolj" + 0.000*"funambulist" + 0.000*"castelli" + 0.000*"temperatu" + '
                           '0.000*"threat" + 0.000*"dartanyian" + 0.000*"maruki" + 0.000*"palmdal" + 0.000*"daughti" + '
                           '0.000*"ifupport"')])

    def test_add_corpus(self):
        train = Train(self.parameters)
        train.add_corpus()
        self.assertEqual(len(train.corpus.dictionary), 184080)
