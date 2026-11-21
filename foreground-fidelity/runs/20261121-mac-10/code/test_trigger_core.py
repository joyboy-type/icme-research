import unittest
from trigger_core import route

class TriggerTests(unittest.TestCase):
 def test_background_advantage_can_override_opposing_subject(self):
  r=route('A','B','eligible',-.25,.3375)
  self.assertEqual((r['winner'],r['triggered'],r['status']),('B',True,'switched'))
 def test_unreliable_mask_keeps_B1_even_with_opposition(self):
  r=route('A','B','unreliable_region',-.25,.3375)
  self.assertEqual((r['winner'],r['status']),('A','unable'))
 def test_explicit_foreground_edit_is_not_overridden(self):
  r=route('A','B','ineligible_scope',-.25,.3375)
  self.assertEqual((r['winner'],r['triggered']),('A',False))
 def test_same_winner_control_is_unchanged_even_with_uncertainty(self):
  r=route('B','B','uncertain_scene',-.1,-.2)
  self.assertEqual((r['winner'],r['status']),('B','unchanged'))
 def test_nonfinite_or_numerically_negligible_evidence_cannot_trigger(self):
  for fg,bg in [(float('nan'),.4),(-1e-9,.4),(-.4,.2)]:
   r=route('A','B','eligible',fg,bg)
   self.assertFalse(r['triggered']);self.assertEqual(r['winner'],'A')

if __name__=='__main__':unittest.main()
