import unittest
import features
import pdb

class TestFeatures(unittest.TestCase):
    
    def setUp(self):
        pass


    def test_chunks(self):
        sents = ['[NP A_DT 26-year-old_JJ mentally_RB challenged_VBN youth_NN ] [VP died_VBD ] [PP at_IN ] [NP a_DT birthday_NN party_NN ] [SBAR after_IN ] [NP he_PRP ] [VP was_VBD       brutally_RB assaulted_VBN ] [PP by_IN ] [NP five_CD members_NNS ] [PP of_IN ] [NP a_DT family_NN ] [PP after_IN ] [NP a_DT woman_NN ] [PP of_IN ] [NP the_DT famil      y_NN ] [VP thought_VBD ] [NP he_PRP ] [VP was_VBD harassing_VBG ] [NP her_PRP]',\
                 '[NP I_PRP ] [VP want_VBP ] [NP that_DT File_NNP ] [ADVP immediately_RB ] ,_, [NP it_PRP ] [VP will_MD be_VB ] [ADVP there_RB ] [PP in_IN ] [NP cupboard_NN ] ,_, [VP please_VB go_VB get_VB ] [NP it_PRP ] [ADVP right_RB now_RB]']
        ans = [features.do_chunk(sent) for sent in sents]
        self.assertEquals(ans, [0, 1])

    def test_call(self):
        sents = ['call me now', 'plz do not call me now', 'bring your bag',
                    'not bring your bag', 'im not coming to the meeting, but please bring my bag', 'Don\'t call me']
        ans = [features.call(s) for s in sents]
        self.assertEquals(ans, [1, 0, 1, 0, 1, 0])

    def test_fire(self):
        sents = ['our house caught fire', 'he was not injured']
        ans = [features.fire(s) for s in sents]
        self.assertEquals(ans, [1, 0])

    def test_health(self):
        sents = ['i need an ambulance', 'do not call an ambulance', 'don\'t call an ambulance']
        ans = [features.health(s) for s in sents]
        self.assertEquals(ans, [1, 0, 0])

    def test_meet_suggest(self):
        sents = ["i don't have a meeting at 5PM today", "i have a meeting at 5PM today"]
        ans = [features.meet_suggest(s) for s in sents]
        self.assertEquals(ans, [0,1])
    
    def test_date(self):
        sents = ["i don't have a meeting at 5PM today", "i have a meeting at 5PM today"]
        ans = [features.date(s) for s in sents]
        self.assertEquals(ans, [1,1])

if __name__ == '__main__':
    unittest.main()
