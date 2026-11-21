import unittest
from merge import merge

def row(text,y,x=100):return {'text':text,'box':[x,y,x+240,y+15]}
UP='The upper neighboring paragraph line'
DOWN='The lower neighboring paragraph line'
MISSING='MISSING TEXT SAMPLE SEVEN'
class MergeTest(unittest.TestCase):
    def setUp(self):
        self.base=[row(UP,100),row(DOWN,160)]
        self.donor=[row(UP,100),row(MISSING,130),row(DOWN,160)]
    def test_supported_gap_is_filled_between_anchors_without_editing_originals(self):
        output=merge(self.base,self.donor,1000,1000,True)
        self.assertEqual([r['text'] for r in output['lines']],[UP,MISSING,DOWN])
        self.assertEqual([r['text'] for r in output['lines'] if r['origin']=='receiver'],[UP,DOWN])
        self.assertEqual(len(output['insertions']),1)
    def test_occupied_but_misrecognized_row_is_rejected(self):
        base=[self.base[0],row('A completely garbled existing reading',130),self.base[1]]
        output=merge(base,self.donor,1000,1000,True)
        self.assertEqual(output['insertions'],[])
        self.assertTrue(any('occupied' in r['reason'] for r in output['decisions']))
    def test_text_only_merge_accepts_but_geometry_rejects_wrong_column(self):
        donor=[row(UP,100,500),row(MISSING,130,500),row(DOWN,160,500)]
        self.assertEqual(len(merge(self.base,donor,1000,1000,False)['insertions']),1)
        self.assertEqual(merge(self.base,donor,1000,1000,True)['insertions'],[])
    def test_one_sided_anchor_is_insufficient(self):
        output=merge(self.base,self.donor[:2],1000,1000,True)
        self.assertEqual(output['insertions'],[])
    def test_donor_repetition_does_not_duplicate_the_insertion(self):
        donor=[self.donor[0],self.donor[1],self.donor[1],self.donor[2]]
        output=merge(self.base,donor,1000,1000,True)
        self.assertEqual([r['text'] for r in output['lines']].count(MISSING),1)

if __name__=='__main__':unittest.main()
