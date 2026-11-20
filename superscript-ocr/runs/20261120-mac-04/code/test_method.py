"""Targeted checks: prevent baseline digits, split groups, budget and anchor errors."""
import unittest
import numpy as np
import cv2
try:
    from method import proposals, paired_jobs, attach, numeric
except ImportError:
    proposals = paired_jobs = attach = numeric = None


class MethodTest(unittest.TestCase):
    def test_small_raised_group_is_found_without_gold_and_baseline_is_rejected(self):
        self.assertIsNotNone(proposals, 'automatic proposal method is missing')
        image = np.full((2000, 400), 255, np.uint8)
        for x in range(30, 220, 18):
            cv2.rectangle(image, (x, 90), (x+10, 110), 0, -1)
        cv2.rectangle(image, (239, 80), (244, 90), 0, -1)
        cv2.rectangle(image, (248, 80), (253, 90), 0, -1)
        lines = [{'box': [.07, .04, .70, .016], 'text': 'body text with a superscript'}]
        found = proposals(image, lines)
        self.assertTrue(any(p['box'][0] <= 239 and p['box'][2] >= 254 for p in found))
        self.assertFalse(any(40 <= (p['box'][0]+p['box'][2])/2 <= 220 for p in found))

    def test_uniform_has_identical_per_call_pixel_budget_and_stays_in_page(self):
        self.assertIsNotNone(paired_jobs, 'matched-budget construction is missing')
        pairs = paired_jobs([{'box': [0, 0, 12, 14], 'line_id': 0}], 200, 160)
        self.assertEqual(len(pairs), 2)
        a, b = pairs
        self.assertEqual(a['box'][2]-a['box'][0], b['box'][2]-b['box'][0])
        self.assertEqual(a['box'][3]-a['box'][1], b['box'][3]-b['box'][1])
        self.assertTrue(0 <= b['box'][0] < b['box'][2] <= 200)

    def test_attachment_uses_position_and_does_not_duplicate_existing_marker(self):
        self.assertIsNotNone(attach, 'position-based attachment is missing')
        lines = [{'text': 'Alpha beta.', 'box': [0, 0, 100, 20],
                  'chars': [{'text': c, 'start': i, 'end': i+1, 'box': [i*8, 0, 7, 20]}
                            for i, c in enumerate('Alpha beta.')]}]
        p = {'text': '26', 'xy': [92, 3], 'box': [87, 0, 97, 8]}
        patches = attach(lines, [p], [])
        self.assertEqual(patches[0]['offset'], 11)
        self.assertEqual(patches[0]['left'], 'Alpha beta.')
        self.assertEqual(attach(lines, [p], [p]), [])
        self.assertEqual(numeric('[２６]'), '26')

    def test_word_level_boxes_use_word_end_and_abstain_inside_body_words(self):
        self.assertIsNotNone(attach, 'position-based attachment is missing')
        text='Alpha beta.'
        chars=[dict(text=c,start=i,end=i+1,box=[0,0,40,20] if i<5 else [48,0,40,20])
               for i,c in enumerate(text)]
        lines=[dict(text=text,box=[0,0,100,20],chars=chars)]
        p=dict(text='26',xy=[92,3],box=[87,0,97,8])
        self.assertEqual(attach(lines,[p],[])[0]['offset'],11)
        fragment=dict(text='1',xy=[60,3],box=[58,0,62,8])
        self.assertEqual(attach(lines,[fragment],[]),[])

if __name__ == '__main__':
    unittest.main()
