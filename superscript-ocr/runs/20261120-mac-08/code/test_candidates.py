import unittest
import numpy as np
import cv2
try:
    from candidates import isolate, rerank, render
except ImportError:
    isolate=rerank=render=None

class CandidateTest(unittest.TestCase):
    def test_partial_baseline_glyph_and_small_leading_dot_are_removed(self):
        self.assertIsNotNone(isolate,'component isolation missing')
        gray=np.full((120,200),255,np.uint8)
        cv2.rectangle(gray,(35,45),(42,70),0,-1) # body glyph crosses proposal bottom
        cv2.rectangle(gray,(50,43),(52,45),0,-1) # dot before raised numeral
        cv2.rectangle(gray,(60,43),(65,53),0,-1)
        count,cc,stats,_=cv2.connectedComponentsWithStats((gray<180).astype('uint8'))
        proposal=dict(box=[33,40,68,56],line_height=25,baseline=68,score=1,line_id=0)
        clean=isolate(proposal,cc,stats)
        self.assertIsNotNone(clean)
        self.assertEqual(clean['box'],[59,42,67,55])
        rendered=render(gray,cc,clean)
        self.assertTrue(np.all(rendered[:,0]==255))
        self.assertLess(rendered.min(),1)

    def test_a_small_connector_between_digits_is_kept(self):
        self.assertIsNotNone(isolate,'component isolation missing')
        gray=np.full((120,200),255,np.uint8)
        cv2.rectangle(gray,(60,43),(64,53),0,-1)
        cv2.rectangle(gray,(72,43),(77,53),0,-1)
        cv2.rectangle(gray,(67,48),(69,49),0,-1)
        _,cc,stats,_=cv2.connectedComponentsWithStats((gray<180).astype('uint8'))
        clean=isolate(dict(box=[58,40,80,57],line_height=25,baseline=68,score=1,line_id=0),cc,stats)
        rendered=render(gray,cc,clean);x0,y0,_,_=clean['box']
        self.assertEqual(int(rendered[48-y0,68-x0]),0)

    def test_real_sized_raised_group_outranks_a_tiny_dot_and_cap_is_enforced(self):
        self.assertIsNotNone(rerank,'geometric reranking missing')
        gray=np.full((120,200),255,np.uint8)
        cv2.rectangle(gray,(20,42),(23,45),0,-1)
        cv2.rectangle(gray,(70,45),(75,56),0,-1)
        _,cc,stats,_=cv2.connectedComponentsWithStats((gray<180).astype('uint8'))
        pool=[dict(box=[18,40,26,48],line_height=28,baseline=75,score=9,line_id=0),
              dict(box=[68,43,78,59],line_height=28,baseline=75,score=1,line_id=0)]
        ranked=rerank(pool,cc,stats,limit=1)
        self.assertEqual(len(ranked),1)
        self.assertGreater(ranked[0]['box'][0],60)

if __name__=='__main__':unittest.main()
