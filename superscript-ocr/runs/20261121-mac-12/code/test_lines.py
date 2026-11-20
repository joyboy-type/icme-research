import unittest
from lines import physical_lines, compare_line


class LineTest(unittest.TestCase):
    def test_same_baseline_fragments_merge_but_next_line_stays_separate(self):
        spans=[dict(category_type='text_span',text='alpha beta',poly=[0,0,40,0,40,10,0,10]),dict(category_type='text_span',text='gamma delta',poly=[45,0,95,0,95,10,45,10]),dict(category_type='text_span',text='one two three four',poly=[0,20,95,20,95,30,0,30])]
        result=physical_lines(spans)
        self.assertEqual([r['text'] for r in result],['alpha beta gamma delta','one two three four'])

    def test_correct_text_is_present_despite_a_merged_observation_box(self):
        line=dict(text='alpha beta gamma delta',box=[10,30,100,40])
        observations=[dict(text='earlier words alpha beta gamma delta later words',box=[0,0,120,80],confidence=1)]
        self.assertEqual(compare_line(line,observations)['screen_status'],'present')

    def test_empty_source_position_is_not_the_same_as_a_garbled_reading(self):
        line=dict(text='alpha beta gamma delta',box=[10,30,100,40])
        self.assertEqual(compare_line(line,[])['screen_status'],'omission_candidate')
        observed=[dict(text='xxxx yyyy zzzz',box=[10,30,100,40],confidence=.3)]
        self.assertEqual(compare_line(line,observed)['screen_status'],'misread_or_merged_omission_candidate')


if __name__=='__main__':unittest.main()
