import unittest
try:
    from score import at_marker, anchor_ok
except ImportError:
    at_marker = anchor_ok = None

class ScoreTest(unittest.TestCase):
    def test_same_number_at_a_different_position_is_not_a_hit(self):
        self.assertIsNotNone(at_marker, 'position scorer missing')
        m={'identity':'[26]','span':{'poly':[100,100,120,100,120,120,100,120]}}
        self.assertTrue(at_marker({'text':'26','xy':[110,110]},m))
        self.assertFalse(at_marker({'text':'26','xy':[300,110]},m))
        self.assertFalse(at_marker({'text':'2','xy':[110,110]},m))

    def test_correct_number_with_wrong_clause_is_not_an_anchor_hit(self):
        self.assertIsNotNone(anchor_ok, 'anchor scorer missing')
        m={'anchor_before':'The experiment was established.', 'anchor_after':'Next we isolate a clone.'}
        self.assertTrue(anchor_ok({'left':'was established.', 'right':' Next we isolate'},m))
        self.assertFalse(anchor_ok({'left':'this was not observed.', 'right':' Another experiment'},m))

if __name__=='__main__':unittest.main()
