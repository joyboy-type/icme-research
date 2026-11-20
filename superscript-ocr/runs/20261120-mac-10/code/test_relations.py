import unittest
from relations import citations, admit, read_bibliography
from gold import jats_relations
from evaluate import correctness


class RelationTest(unittest.TestCase):
    def test_correct_number_and_anchor_do_not_hide_a_wrong_bibliography_target(self):
        gold=dict(numbers=[3],box=[10,10,20,20])
        candidate=dict(numbers=[3],xy=[15,15],targets=[{'text':'Gold mold surveillance'}],source='doc')
        audit={'doc':{'3':dict(identity_correct=False,boundary_clean=False,actual_leading_reference=23)}}
        result=correctness(candidate,gold,audit)
        self.assertTrue(result['number_correct'])
        self.assertTrue(result['anchor_correct'])
        self.assertFalse(result['all_three_correct'])
        self.assertTrue(result['wrong_link'])

    def test_reference_number_in_its_own_ocr_observation_starts_an_entry(self):
        result=read_bibliography([('p1','References'),('p1','1. Smith study.'),('p1','2.'),('p1','Jones second study.')])
        self.assertEqual(result[2]['text'],'Jones second study.')
        self.assertEqual(result[1]['text'],'Smith study.')

    def test_ranges_expand_without_reading_years_as_citations(self):
        self.assertEqual([c['numbers'] for c in citations('In 2025, work [2, 4–6] improved.')], [[2, 4, 5, 6]])

    def test_valid_bibliography_id_does_not_prove_the_right_number(self):
        candidate = dict(numbers=[7], left='prior studies', right='show gains')
        self.assertTrue(admit(candidate, {3: 'Smith study', 7: 'Jones study'},
                              'prior studies', 'show gains', True))

    def test_wrong_body_anchor_is_rejected_even_with_valid_id(self):
        candidate = dict(numbers=[7], left='prior studies', right='show gains')
        self.assertFalse(admit(candidate, {7: 'Jones study'},
                               'different paragraph', 'unrelated content', True))

    def test_jats_uses_rid_and_rejects_dangling_targets(self):
        xml = '<article><body><p>Prior <xref ref-type="bibr" rid="r7">7</xref> work.</p><p>Bad <xref ref-type="bibr" rid="absent">3</xref>.</p></body><back><ref-list><ref id="r7"><label>7</label><mixed-citation>Jones study</mixed-citation></ref></ref-list></back></article>'
        rows, rejected = jats_relations(xml)
        self.assertEqual([(r['numbers'], r['rids']) for r in rows], [([7], ['r7'])])
        self.assertEqual(rejected[0]['reason'], 'unresolved_rid')

    def test_adjacent_jats_range_is_one_visible_group_with_expanded_targets(self):
        xml = '<article><body><p>Work [<xref ref-type="bibr" rid="b1">1</xref>–<xref ref-type="bibr" rid="b3">3</xref>] agrees.</p></body><back><ref-list><ref id="b1"><label>1</label></ref><ref id="b2"><label>2</label></ref><ref id="b3"><label>3</label></ref></ref-list></back></article>'
        rows, rejected = jats_relations(xml)
        self.assertEqual([(r['numbers'], r['rids']) for r in rows], [([1, 2, 3], ['b1', 'b2', 'b3'])])
        self.assertEqual(rejected, [])


if __name__ == '__main__':
    unittest.main()
