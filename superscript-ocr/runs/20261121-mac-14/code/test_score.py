import unittest
from score import score_position
class ScoreTest(unittest.TestCase):
    def setUp(self):
        self.gold={'text':'MISSING TEXT SAMPLE SEVEN','box':[100,130,340,145]}
        self.base=[{'text':'Upper neighbor line','box':[100,100,340,115]},{'text':'Lower neighbor line','box':[100,160,340,175]}]
        self.add=dict(self.gold,origin='donor',insertion_slot=1)
    def test_recovery_requires_correct_text_at_source_and_between_neighbors(self):
        got=score_position(self.gold,self.base,[self.add], [self.add])
        self.assertEqual(got['correct_recoveries'],1)
        self.assertEqual(got['wrong_position_insertions'],0)
    def test_same_text_in_wrong_column_is_not_a_recovery(self):
        bad=dict(self.add,box=[500,130,740,145])
        got=score_position(self.gold,self.base,[bad],[bad])
        self.assertEqual(got['correct_recoveries'],0)
        self.assertEqual(got['wrong_position_insertions'],1)
    def test_repeated_correct_line_is_counted_as_duplicate(self):
        got=score_position(self.gold,[self.gold],[dict(self.gold,origin='receiver'),self.add],[self.add])
        self.assertEqual(got['duplicate_copies'],1)
    def test_wrong_text_inserted_at_verified_row_is_counted(self):
        bad=dict(self.add,text='INVENTED CONTENT NOT ON THIS SOURCE')
        got=score_position(self.gold,self.base,[bad],[bad])
        self.assertEqual(got['false_insertions'],1)
    def test_correct_box_but_wrong_reading_order_is_not_correct_recovery(self):
        bad=dict(self.add,insertion_slot=0)
        got=score_position(self.gold,self.base,[bad],[bad])
        self.assertEqual(got['correct_recoveries'],0)
        self.assertEqual(got['wrong_position_insertions'],1)
if __name__=='__main__':unittest.main()
