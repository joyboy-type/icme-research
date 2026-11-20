import unittest
from omissions import tokens, common_deletions, normal_looking, footnote_tokens, anchored_gap


class OmissionTest(unittest.TestCase):
    def test_angstrom_notation_is_not_a_missing_letter(self):
        self.assertEqual(tokens(r'7 \overset{\circ}{A}'),tokens('7 Å'))
        self.assertEqual(tokens(r'7 \AA'),tokens('7 Å'))

    def test_spaced_digits_in_superscript_are_one_printed_marker(self):
        self.assertEqual(footnote_tokens('$ ^{2 6} $'),['26'])
        self.assertEqual(footnote_tokens('$ ^{2 8-3 2} $'),['28','32'])

    def test_anchor_gap_distinguishes_omission_from_partial_recognition(self):
        left=['was','previously','established'];right=['this','cell','line']
        self.assertEqual(anchored_gap(left,right,tokens('was previously established this cell line')),[])
        self.assertEqual(anchored_gap(left,right,tokens('was previously established 2 this cell line')),['2'])

    def test_common_deletion_requires_the_same_source_position_in_all_models(self):
        gold='alpha beta one two three four five six seven eight gamma delta'
        missing='alpha beta gamma delta'
        self.assertEqual(common_deletions(tokens(gold),[tokens(missing)]*3),[(2,10)])
        self.assertEqual(common_deletions(tokens(gold),[tokens(missing),tokens(gold),tokens(missing)]),[])

    def test_substitution_is_not_counted_as_omission(self):
        self.assertEqual(common_deletions(tokens('alpha value beta'),[tokens('alpha other beta')]*3),[])

    def test_markup_does_not_turn_coordinates_into_document_text(self):
        self.assertEqual(tokens('<style>x123</style><p data-bbox="1 2 3 4">Dose <sup>12</sup> mg</p>'),['dose','12','mg'])

    def test_repeated_decode_is_not_normal_looking(self):
        self.assertFalse(normal_looking(tokens('a b c d e f g h i j '*50)))
        self.assertTrue(normal_looking(tokens('This is a short complete document with useful content.')))


if __name__=='__main__':unittest.main()
