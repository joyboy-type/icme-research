import unittest
try:
    from diagnose import ink_coverage
except ImportError:
    ink_coverage=None

class CoverageTest(unittest.TestCase):
    def test_a_center_covered_box_can_cut_off_a_digit_group(self):
        self.assertIsNotNone(ink_coverage,'ink coverage diagnostic missing')
        ink=[(10,10),(10,11),(20,10),(20,11)]
        self.assertEqual(ink_coverage(ink,[8,8,23,14]),1.)
        self.assertEqual(ink_coverage(ink,[8,8,17,14]),.5)
        self.assertEqual(ink_coverage(ink,[12,8,18,14]),0.)

    def test_crop_upper_bound_is_exclusive(self):
        self.assertIsNotNone(ink_coverage,'ink coverage diagnostic missing')
        self.assertEqual(ink_coverage([(10,10),(11,10)],[9,9,11,11]),.5)

if __name__=='__main__':unittest.main()
