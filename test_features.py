import unittest
import features
import pdb

class TestFeatures(unittest.TestCase):
    
    def setUp(self):
        pass


    def test_parser(self):
        sents = ['please call me once you are free', 'you should\
                               call me', 'ok']
        ans = [features.parser(s) for s in sents]
        self.assertEquals(ans, [1, 0, 0])

if __name__ == '__main__':
    unittest.main()
