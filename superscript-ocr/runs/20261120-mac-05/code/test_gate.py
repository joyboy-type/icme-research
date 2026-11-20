import unittest
try:
    from gate import decide, context_box
except ImportError:
    decide = context_box = None

class GateTest(unittest.TestCase):
    def test_context_confirms_a_sentence_citation_and_rejects_missing_digit(self):
        self.assertIsNotNone(decide, 'context gate is missing')
        p=dict(text='17',left='the peroneal nerve).',right=' A nerve can be compressed.')
        self.assertTrue(decide(p,'the peroneal nerve).17 A nerve can be compressed.')[0])
        self.assertFalse(decide(p,'the peroneal nerve). A nerve can be compressed.')[0])

    def test_matching_math_exponent_is_still_rejected(self):
        self.assertIsNotNone(decide, 'context gate is missing')
        p=dict(text='2',left='less than 1.8 GeV/c',right='. The decay')
        self.assertFalse(decide(p,'less than 1.8 GeV/c2. The decay')[0])
        isotope=dict(text='13',left='solid state',right=' C NMR spectra')
        self.assertFalse(decide(isotope,'solid state 13C NMR spectra')[0])

    def test_context_disagreement_and_an_unrelated_occurrence_do_not_pass(self):
        self.assertIsNotNone(decide, 'context gate is missing')
        p=dict(text='9',left='developed previously',right=' using PEG')
        self.assertFalse(decide(p,'developed previously6 using PEG')[0])
        self.assertFalse(decide(p,'reference9 and a different sentence.')[0])

    def test_bracketed_reference_can_follow_a_chemical_formula(self):
        self.assertIsNotNone(decide, 'context gate is missing')
        p=dict(text='11',left='and CIC(O)SBr.',right=' The unambiguous identification')
        self.assertTrue(decide(p,'and CIC(O)SBr.[11] The unambiguous identification')[0])

    def test_names_without_reference_format_abstain(self):
        self.assertIsNotNone(decide, 'context gate is missing')
        p=dict(text='2',left='S Ding',right=', S Dunn')
        self.assertFalse(decide(p,'S Ding2, S Dunn')[0])

    def test_context_dimensions_are_fixed_for_the_same_crop_shape_at_page_edges(self):
        self.assertIsNotNone(context_box, 'context crop construction missing')
        self.assertEqual(context_box([0,0,10,15],500,800)[2:],context_box([400,700,410,715],500,800)[2:])

if __name__=='__main__':unittest.main()
